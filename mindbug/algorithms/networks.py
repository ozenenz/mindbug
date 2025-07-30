from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MindbugNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.norm_layers.append(nn.LayerNorm(hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_proj(x))
        for layer, norm in zip(self.hidden_layers, self.norm_layers):
            residual = x
            x = F.relu(layer(x))
            x = norm(x + residual)
        return self.output_layer(x)


class MindbugStateEncoder:
    def __init__(self, num_card_types: int = 32):
        self.num_card_types = num_card_types
        self.card_name_to_idx = {}
        self._initialize_card_indices()

    def _initialize_card_indices(self):
        from ..game.cards import CardDefinitions

        cards = CardDefinitions.get_first_contact_cards()
        for i, name in enumerate(sorted(cards.keys())):
            self.card_name_to_idx[name] = i

    def encode_state(self, state, player) -> torch.Tensor:
        features = []
        features.extend(
            [
                state.life[player] / 10.0,
                state.life[player.other()] / 10.0,
                state.mindbugs_available[player] / 2.0,
                state.mindbugs_available[player.other()] / 2.0,
                float(state.current_player == player),
                float(state.extra_turn_pending == player),
                float(state.mindbug_decision_pending),
                float(state.hunter_choice_pending),
            ]
        )
        hand_encoding = [0.0] * self.num_card_types
        for card in state.hands[player]:
            card_idx = self._get_card_index(card)
            hand_encoding[card_idx] = min(hand_encoding[card_idx] + 0.2, 1.0)
        features.extend(hand_encoding)
        my_creatures = [0.0] * self.num_card_types
        for creature in state.play_areas[player]:
            card_idx = self._get_card_index(creature.card)
            my_creatures[card_idx] = min(my_creatures[card_idx] + 0.5, 1.0)
        features.extend(my_creatures)
        my_exhausted = [0.0] * self.num_card_types
        for creature in state.play_areas[player]:
            if creature.is_exhausted:
                card_idx = self._get_card_index(creature.card)
                my_exhausted[card_idx] = 1.0
        features.extend(my_exhausted)
        opp_creatures = [0.0] * self.num_card_types
        for creature in state.play_areas[player.other()]:
            card_idx = self._get_card_index(creature.card)
            opp_creatures[card_idx] = min(opp_creatures[card_idx] + 0.5, 1.0)
        features.extend(opp_creatures)
        opp_exhausted = [0.0] * self.num_card_types
        for creature in state.play_areas[player.other()]:
            if creature.is_exhausted:
                card_idx = self._get_card_index(creature.card)
                opp_exhausted[card_idx] = 1.0
        features.extend(opp_exhausted)
        features.extend(
            [
                len(state.decks[player]) / 10.0,
                len(state.decks[player.other()]) / 10.0,
                len(state.discard_piles[player]) / 20.0,
                len(state.discard_piles[player.other()]) / 20.0,
            ]
        )
        pending_encoding = [0.0] * self.num_card_types
        if state.mindbug_decision_pending and state.pending_creature_play:
            _, pending_card = state.pending_creature_play
            card_idx = self._get_card_index(pending_card)
            pending_encoding[card_idx] = 1.0
        features.extend(pending_encoding)
        blocker_encoding = [0.0] * 10
        if state.hunter_choice_pending:
            for idx in state.valid_blockers:
                if idx < 10:
                    blocker_encoding[idx] = 1.0
        features.extend(blocker_encoding)
        attacking_encoding = [0.0] * self.num_card_types
        if state.attacking_creature:
            card_idx = self._get_card_index(state.attacking_creature.card)
            attacking_encoding[card_idx] = 1.0
        features.extend(attacking_encoding)
        return torch.tensor(features, dtype=torch.float32)

    def encode_action(self, action, state, player) -> torch.Tensor:
        features = []
        action_type_encoding = [0.0] * 5
        action_type_encoding[action.action_type.value - 1] = 1.0
        features.extend(action_type_encoding)
        card_encoding = [0.0] * self.num_card_types
        if action.card:
            card_idx = self._get_card_index(action.card)
            card_encoding[card_idx] = 1.0
        features.extend(card_encoding)
        if action.creature_index is not None:
            features.append(action.creature_index / 10.0)
        else:
            features.append(0.0)
        if action.target_index is not None:
            features.append(action.target_index / 10.0)
        else:
            features.append(0.0)
        return torch.tensor(features, dtype=torch.float32)

    def _get_card_index(self, card) -> int:
        return self.card_name_to_idx.get(card.name, 0)

    def get_state_dim(self) -> int:
        return 8 + 7 * self.num_card_types + 4 + 10

    def get_action_dim(self) -> int:
        return 5 + self.num_card_types + 2
