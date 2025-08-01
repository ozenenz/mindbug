"""Deep Counterfactual Regret Minimization implementation."""

import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..core import Action, GameState, MindbugEngine, Player
from .networks import DualBranchNetwork, StateEncoder


@dataclass
class CFRSample:
    """Single training sample for CFR networks."""

    card_indices: torch.Tensor
    history_features: torch.Tensor
    value: float  # Regret for advantage, probability for strategy
    weight: float  # Linear CFR weight
    iteration: int


class ReservoirBuffer:
    """Reservoir sampling buffer for memory-efficient storage."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[CFRSample] = []
        self.num_added = 0

    def add(self, sample: CFRSample) -> None:
        """Add sample using reservoir sampling."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            # Reservoir sampling
            idx = random.randint(0, self.num_added)
            if idx < self.capacity:
                self.buffer[idx] = sample
        self.num_added += 1

    def sample_batch(self, batch_size: int) -> List[CFRSample]:
        """Sample a batch from the buffer."""
        if len(self.buffer) <= batch_size:
            return self.buffer.copy()
        return random.sample(self.buffer, batch_size)

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.num_added = 0

    def __len__(self) -> int:
        return len(self.buffer)


class DeepCFR:
    """Deep Counterfactual Regret Minimization algorithm."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.get("use_gpu", True) else "cpu"
        )

        # Initialize encoder
        self.encoder = StateEncoder(num_card_types=config.get("num_card_types", 32))

        # Initialize buffers
        buffer_size = config.get("buffer_size", 2000000)
        self.advantage_buffers = {
            Player.PLAYER_1: ReservoirBuffer(buffer_size),
            Player.PLAYER_2: ReservoirBuffer(buffer_size),
        }
        self.strategy_buffers = {
            Player.PLAYER_1: ReservoirBuffer(buffer_size),
            Player.PLAYER_2: ReservoirBuffer(buffer_size),
        }

        # Algorithm parameters
        self.use_linear_cfr = config.get("use_linear_cfr", True)
        self.iteration_count = 0

        # Networks (created fresh each iteration)
        self.value_networks: Dict[Player, DualBranchNetwork] = {}
        self.strategy_network: Optional[DualBranchNetwork] = None

    def train(self, num_iterations: int) -> None:
        """Train Deep CFR for specified iterations."""
        print(f"Starting Deep CFR training for {num_iterations} iterations...")
        print(f"Using device: {self.device}")

        for iteration in range(1, num_iterations + 1):
            self.iteration_count = iteration

            # Create fresh value networks
            self.value_networks = {
                Player.PLAYER_1: self._create_network(),
                Player.PLAYER_2: self._create_network(),
            }

            # Alternate traversing player
            traverser = Player.PLAYER_1 if iteration % 2 == 1 else Player.PLAYER_2

            # Perform traversals
            num_traversals = self.config.get("traversals_per_iteration", 1000)
            for _ in range(num_traversals):
                state = MindbugEngine.create_initial_state()
                self._traverse(state, traverser, iteration)

            # Train value networks
            for player in [Player.PLAYER_1, Player.PLAYER_2]:
                self._train_value_network(player, iteration)

            # Periodically train strategy network
            if iteration % self.config.get("strategy_interval", 10) == 0:
                self._train_strategy_network(iteration)

            # Logging
            if iteration % self.config.get("log_interval", 100) == 0:
                self._log_progress(iteration)

    def _create_network(self) -> DualBranchNetwork:
        """Create a fresh network."""
        return DualBranchNetwork(
            card_embedding_dim=self.config.get("card_embedding_dim", 128),
            history_dim=self.encoder.history_dim,
            hidden_dim=self.config.get("hidden_dim", 256),
            num_card_types=self.config.get("num_card_types", 32),
        ).to(self.device)

    def _traverse(self, state: GameState, traverser: Player, iteration: int) -> float:
        """External sampling Monte Carlo CFR traversal."""
        if state.is_terminal():
            winner = state.get_winner()
            if winner is None:
                return 0.0
            return 1.0 if winner == traverser else -1.0

        legal_actions = state.get_legal_actions()
        if not legal_actions:
            # No actions = loss
            return -1.0 if state.current_player == traverser else 1.0

        current_player = state.current_player

        if current_player == traverser:
            # Traversing player: compute regrets
            strategy = self._get_strategy(state, legal_actions, current_player, iteration)
            action_values = {}

            # Get value for each action
            for action in legal_actions:
                new_state = MindbugEngine.apply_action(state, action)
                action_values[action] = self._traverse(new_state, traverser, iteration)

            # Compute counterfactual value
            state_value = sum(strategy[a] * action_values[a] for a in legal_actions)

            # Store regret samples
            for action in legal_actions:
                regret = action_values[action] - state_value

                # Encode state-action
                card_indices, history_features = self.encoder.encode_state_and_action(
                    state, current_player, action
                )

                # Linear CFR weight
                weight = iteration if self.use_linear_cfr else 1

                sample = CFRSample(
                    card_indices=card_indices,
                    history_features=history_features,
                    value=regret,
                    weight=weight,
                    iteration=iteration,
                )
                self.advantage_buffers[current_player].add(sample)

            return state_value
        else:
            # Opponent: sample action and store strategy
            strategy = self._get_strategy(state, legal_actions, current_player, iteration)

            # Store strategy samples
            for action in legal_actions:
                card_indices, history_features = self.encoder.encode_state_and_action(
                    state, current_player, action
                )

                weight = iteration if self.use_linear_cfr else 1

                sample = CFRSample(
                    card_indices=card_indices,
                    history_features=history_features,
                    value=strategy[action],
                    weight=weight,
                    iteration=iteration,
                )
                self.strategy_buffers[current_player].add(sample)

            # Sample action
            action = self._sample_action(legal_actions, strategy)
            new_state = MindbugEngine.apply_action(state, action)
            return self._traverse(new_state, traverser, iteration)

    def _get_strategy(
        self, state: GameState, legal_actions: List[Action], player: Player, iteration: int
    ) -> Dict[Action, float]:
        """Compute strategy using regret matching."""
        if player not in self.value_networks:
            # Uniform if no network
            return {a: 1.0 / len(legal_actions) for a in legal_actions}

        # Get advantages from value network
        network = self.value_networks[player]
        network.eval()

        advantages = {}
        with torch.no_grad():
            for action in legal_actions:
                card_indices, history_features = self.encoder.encode_state_and_action(
                    state, player, action
                )

                # Add batch dimension and move to device
                card_indices = card_indices.unsqueeze(0).to(self.device)
                history_features = history_features.unsqueeze(0).to(self.device)

                # Get advantage
                advantage = network(card_indices, history_features).item()
                advantages[action] = advantage

        # Regret matching
        positive_regrets = {a: max(0, adv) for a, adv in advantages.items()}
        total = sum(positive_regrets.values())

        if total > 0:
            strategy = {a: reg / total for a, reg in positive_regrets.items()}
        else:
            # Uniform if all regrets negative
            strategy = {a: 1.0 / len(legal_actions) for a in legal_actions}

        return strategy

    def _sample_action(self, actions: List[Action], strategy: Dict[Action, float]) -> Action:
        """Sample action according to strategy."""
        actions_list = list(actions)
        probs = [strategy.get(a, 0.0) for a in actions_list]

        # Normalize
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            probs = [1.0 / len(actions_list)] * len(actions_list)

        return np.random.choice(actions_list, p=probs)

    def _train_value_network(self, player: Player, iteration: int) -> None:
        """Train value network on regret samples."""
        network = self.value_networks[player]
        network.train()

        optimizer = optim.Adam(
            network.parameters(),
            lr=self.config.get("learning_rate", 1e-3),
            weight_decay=self.config.get("weight_decay", 1e-4),
        )

        batch_size = self.config.get("batch_size", 2048)
        num_epochs = self.config.get("value_epochs", 100)

        for epoch in range(num_epochs):
            batch = self.advantage_buffers[player].sample_batch(batch_size)
            if not batch:
                continue

            # Prepare batch
            card_indices = torch.stack([s.card_indices for s in batch]).to(self.device)
            history_features = torch.stack([s.history_features for s in batch]).to(self.device)
            targets = torch.tensor([s.value for s in batch], dtype=torch.float32).to(self.device)

            # Linear CFR weights
            if self.use_linear_cfr:
                weights = torch.tensor(
                    [s.weight / iteration for s in batch], dtype=torch.float32
                ).to(self.device)
            else:
                weights = torch.ones(len(batch), dtype=torch.float32).to(self.device)

            # Forward pass
            predictions = network(card_indices, history_features).squeeze()

            # Weighted MSE loss
            loss = (weights * (predictions - targets) ** 2).mean()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                network.parameters(), max_norm=self.config.get("grad_clip", 1.0)
            )

            optimizer.step()

    def _train_strategy_network(self, iteration: int) -> None:
        """Train strategy network on strategy samples."""
        self.strategy_network = self._create_network()
        self.strategy_network.train()

        # Combine samples from both players
        all_samples = []
        for player in [Player.PLAYER_1, Player.PLAYER_2]:
            samples = self.strategy_buffers[player].sample_batch(
                self.config.get("strategy_batch_size", 10000)
            )
            all_samples.extend(samples)

        if not all_samples:
            return

        optimizer = optim.Adam(
            self.strategy_network.parameters(),
            lr=self.config.get("strategy_learning_rate", 1e-4),
        )

        batch_size = self.config.get("batch_size", 2048)
        num_epochs = self.config.get("strategy_epochs", 200)

        for epoch in range(num_epochs):
            random.shuffle(all_samples)

            for i in range(0, len(all_samples), batch_size):
                batch = all_samples[i : i + batch_size]

                # Prepare batch
                card_indices = torch.stack([s.card_indices for s in batch]).to(self.device)
                history_features = torch.stack([s.history_features for s in batch]).to(self.device)
                targets = torch.tensor([s.value for s in batch], dtype=torch.float32).to(
                    self.device
                )

                # Forward pass
                predictions = self.strategy_network(card_indices, history_features).squeeze()

                # MSE loss for probability approximation
                loss = F.mse_loss(torch.sigmoid(predictions), targets)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def get_policy(self, state: GameState, player: Player) -> Dict[Action, float]:
        """Get final playing policy."""
        legal_actions = state.get_legal_actions()

        if self.strategy_network is not None:
            # Use trained strategy network
            self.strategy_network.eval()
            probs = {}

            with torch.no_grad():
                for action in legal_actions:
                    card_indices, history_features = self.encoder.encode_state_and_action(
                        state, player, action
                    )

                    card_indices = card_indices.unsqueeze(0).to(self.device)
                    history_features = history_features.unsqueeze(0).to(self.device)

                    logit = self.strategy_network(card_indices, history_features).item()
                    probs[action] = torch.sigmoid(torch.tensor(logit)).item()

            # Normalize
            total = sum(probs.values())
            if total > 0:
                return {a: p / total for a, p in probs.items()}

        # Fall back to value network strategy
        return self._get_strategy(state, legal_actions, player, self.iteration_count)

    def _log_progress(self, iteration: int) -> None:
        """Log training progress."""
        buffer_sizes = {
            "P1_adv": len(self.advantage_buffers[Player.PLAYER_1]),
            "P2_adv": len(self.advantage_buffers[Player.PLAYER_2]),
            "P1_str": len(self.strategy_buffers[Player.PLAYER_1]),
            "P2_str": len(self.strategy_buffers[Player.PLAYER_2]),
        }
        print(f"Iteration {iteration} - Buffer sizes: {buffer_sizes}")

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "config": self.config,
            "iteration_count": self.iteration_count,
            "encoder_mapping": self.encoder.card_to_idx,
        }

        if self.strategy_network is not None:
            checkpoint["strategy_network"] = self.strategy_network.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.iteration_count = checkpoint.get("iteration_count", 0)
        self.encoder.card_to_idx = checkpoint.get("encoder_mapping", {})

        if "strategy_network" in checkpoint:
            self.strategy_network = self._create_network()
            self.strategy_network.load_state_dict(checkpoint["strategy_network"])
            self.strategy_network.eval()
