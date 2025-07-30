import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..game.actions import Action
from ..game.constants import Player
from ..game.engine import MindbugEngine
from ..game.state import GameState
from .networks import MindbugNetwork, MindbugStateEncoder


@dataclass
class CFRSample:
    features: torch.Tensor
    value: float
    weight: float
    iteration: int


class ReservoirBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.num_added = 0

    def add(self, sample: CFRSample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            idx = random.randint(0, self.num_added)
            if idx < self.capacity:
                self.buffer[idx] = sample
        self.num_added += 1

    def sample_batch(self, batch_size: int) -> List[CFRSample]:
        if len(self.buffer) <= batch_size:
            return self.buffer.copy()
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer = []
        self.num_added = 0

    def __len__(self):
        return len(self.buffer)


class DeepCFR:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available() and config.get("use_gpu", True)
            else "cpu"
        )
        print(f"Using device: {self.device}")
        self.encoder = MindbugStateEncoder(
            num_card_types=config.get("num_card_types", 32)
        )
        self.state_dim = self.encoder.get_state_dim()
        self.action_dim = self.encoder.get_action_dim()
        self.value_networks = {}
        buffer_size = config.get("buffer_size", 2000000)
        self.advantage_buffers = {
            Player.PLAYER_1: ReservoirBuffer(buffer_size),
            Player.PLAYER_2: ReservoirBuffer(buffer_size),
        }
        self.strategy_buffers = {
            Player.PLAYER_1: ReservoirBuffer(buffer_size),
            Player.PLAYER_2: ReservoirBuffer(buffer_size),
        }
        self.use_linear_cfr = config.get("use_linear_cfr", True)
        self.store_all_networks = config.get("store_all_networks", False)
        self.stored_networks = []
        self.iteration_count = 0

    def _create_network(self) -> MindbugNetwork:
        return MindbugNetwork(
            input_dim=self.state_dim + self.action_dim,
            hidden_dim=self.config.get("hidden_dim", 512),
            num_layers=self.config.get("num_layers", 4),
        ).to(self.device)

    def train(self, num_iterations: int):
        print(f"Starting Deep CFR training for {num_iterations} iterations...")
        for iteration in range(1, num_iterations + 1):
            self.iteration_count = iteration
            self.value_networks = {
                Player.PLAYER_1: self._create_network(),
                Player.PLAYER_2: self._create_network(),
            }
            traverser = Player.PLAYER_1 if iteration % 2 == 1 else Player.PLAYER_2
            num_traversals = self.config.get("traversals_per_iteration", 1000)
            for _ in range(num_traversals):
                state = MindbugEngine.create_initial_state()
                self._traverse(state, traverser, iteration)
            for player in [Player.PLAYER_1, Player.PLAYER_2]:
                self._train_value_network(player, iteration)
            if self.store_all_networks:
                self.stored_networks.append(
                    {
                        "iteration": iteration,
                        Player.PLAYER_1: self.value_networks[
                            Player.PLAYER_1
                        ].state_dict(),
                        Player.PLAYER_2: self.value_networks[
                            Player.PLAYER_2
                        ].state_dict(),
                    }
                )
            if iteration % self.config.get("log_interval", 100) == 0:
                buffer_sizes = {
                    "P1_adv": len(self.advantage_buffers[Player.PLAYER_1]),
                    "P2_adv": len(self.advantage_buffers[Player.PLAYER_2]),
                    "P1_str": len(self.strategy_buffers[Player.PLAYER_1]),
                    "P2_str": len(self.strategy_buffers[Player.PLAYER_2]),
                }
                print(f"Iteration {iteration} - Buffer sizes: {buffer_sizes}")

    def _traverse(self, state: GameState, traverser: Player, iteration: int) -> float:
        if state.is_terminal():
            winner = state.get_winner()
            if winner is None:
                return 0.0
            return 1.0 if winner == traverser else -1.0
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            return -1.0 if state.current_player == traverser else 1.0
        current_player = state.current_player
        if current_player == traverser:
            strategy = self._get_strategy(
                state, legal_actions, current_player, iteration
            )
            action_values = {}
            for action in legal_actions:
                new_state = MindbugEngine.apply_action(state, action)
                action_values[action] = self._traverse(new_state, traverser, iteration)
            state_value = sum(strategy[a] * action_values[a] for a in legal_actions)
            for action in legal_actions:
                advantage = action_values[action] - state_value
                state_encoding = self.encoder.encode_state(state, current_player)
                action_encoding = self.encoder.encode_action(
                    action, state, current_player
                )
                features = torch.cat([state_encoding, action_encoding])
                weight = iteration if self.use_linear_cfr else 1
                sample = CFRSample(
                    features=features,
                    value=advantage,
                    weight=weight,
                    iteration=iteration,
                )
                self.advantage_buffers[current_player].add(sample)
            return state_value
        else:
            strategy = self._get_strategy(
                state, legal_actions, current_player, iteration
            )
            state_encoding = self.encoder.encode_state(state, current_player)
            for action in legal_actions:
                action_encoding = self.encoder.encode_action(
                    action, state, current_player
                )
                features = torch.cat([state_encoding, action_encoding])
                weight = iteration if self.use_linear_cfr else 1
                sample = CFRSample(
                    features=features,
                    value=strategy[action],
                    weight=weight,
                    iteration=iteration,
                )
                self.strategy_buffers[current_player].add(sample)
            action = self._sample_action(legal_actions, strategy)
            new_state = MindbugEngine.apply_action(state, action)
            return self._traverse(new_state, traverser, iteration)

    def _get_strategy(
        self,
        state: GameState,
        legal_actions: List[Action],
        player: Player,
        iteration: int,
    ) -> Dict[Action, float]:
        if player not in self.value_networks:
            return {a: 1.0 / len(legal_actions) for a in legal_actions}
        advantages = {}
        network = self.value_networks[player]
        with torch.no_grad():
            for action in legal_actions:
                state_encoding = self.encoder.encode_state(state, player)
                action_encoding = self.encoder.encode_action(action, state, player)
                features = torch.cat([state_encoding, action_encoding])
                features = features.unsqueeze(0).to(self.device)
                advantage = network(features).item()
                advantages[action] = advantage
        positive_regrets = {a: max(0, adv) for a, adv in advantages.items()}
        total_positive = sum(positive_regrets.values())
        if total_positive > 0:
            strategy = {a: reg / total_positive for a, reg in positive_regrets.items()}
        else:
            strategy = {a: 1.0 / len(legal_actions) for a in legal_actions}
        return strategy

    def _sample_action(
        self, actions: List[Action], strategy: Dict[Action, float]
    ) -> Action:
        actions_list = list(actions)
        probs = [strategy.get(a, 0.0) for a in actions_list]
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            probs = [1.0 / len(actions_list)] * len(actions_list)
        return np.random.choice(actions_list, p=probs)

    def _train_value_network(self, player: Player, iteration: int):
        network = self.value_networks[player]
        optimizer = optim.Adam(
            network.parameters(),
            lr=self.config.get("learning_rate", 1e-3),
            weight_decay=self.config.get("weight_decay", 1e-4),
        )
        batch_size = self.config.get("batch_size", 256)
        num_epochs = self.config.get("num_epochs", 100)
        for epoch in range(num_epochs):
            batch = self.advantage_buffers[player].sample_batch(batch_size)
            if not batch:
                continue
            features = torch.stack([s.features for s in batch]).to(self.device)
            targets = torch.tensor([s.value for s in batch], dtype=torch.float32).to(
                self.device
            )
            if self.use_linear_cfr:
                weights = torch.tensor(
                    [s.weight / iteration for s in batch], dtype=torch.float32
                ).to(self.device)
            else:
                weights = torch.ones(len(batch), dtype=torch.float32).to(self.device)
            predictions = network(features).squeeze()
            loss = (weights * (predictions - targets) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                network.parameters(), max_norm=self.config.get("grad_clip", 1.0)
            )
            optimizer.step()

    def get_policy(self, state: GameState, player: Player) -> Dict[Action, float]:
        legal_actions = state.get_legal_actions()
        if self.store_all_networks and self.stored_networks:
            weights = [net["iteration"] for net in self.stored_networks]
            total_weight = sum(weights)
            probs = [w / total_weight for w in weights]
            selected = np.random.choice(len(self.stored_networks), p=probs)
            network_state = self.stored_networks[selected][player]
            network = self._create_network()
            network.load_state_dict(network_state)
            network.eval()
            self.value_networks[player] = network
            strategy = self._get_strategy(state, legal_actions, player, 1)
            return strategy
        else:
            if player in self.value_networks:
                return self._get_strategy(
                    state, legal_actions, player, self.iteration_count
                )
            else:
                return {a: 1.0 / len(legal_actions) for a in legal_actions}

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "config": self.config,
            "iteration_count": self.iteration_count,
            "encoder_card_mapping": self.encoder.card_name_to_idx,
        }
        if self.store_all_networks:
            checkpoint["stored_networks"] = self.stored_networks
        if hasattr(self, "value_networks") and self.value_networks:
            checkpoint["current_networks"] = {
                player.name: network.state_dict()
                for player, network in self.value_networks.items()
            }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.iteration_count = checkpoint.get("iteration_count", 0)
        self.encoder.card_name_to_idx = checkpoint.get("encoder_card_mapping", {})
        if "stored_networks" in checkpoint:
            self.stored_networks = checkpoint["stored_networks"]
            self.store_all_networks = True
        if "current_networks" in checkpoint:
            self.value_networks = {}
            for player in [Player.PLAYER_1, Player.PLAYER_2]:
                if player.name in checkpoint["current_networks"]:
                    network = self._create_network()
                    network.load_state_dict(checkpoint["current_networks"][player.name])
                    self.value_networks[player] = network
        print(f"Loaded checkpoint from {path}")
