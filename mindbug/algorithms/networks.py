"""Neural network architectures for Deep CFR."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import Action, ActionType, GameState, Player


class DualBranchNetwork(nn.Module):
    """7-layer dual-branch architecture as specified in Deep CFR paper."""

    def __init__(
        self,
        card_embedding_dim: int = 128,
        history_dim: int = 64,
        hidden_dim: int = 256,
        num_card_types: int = 32,
    ):
        super().__init__()

        # Card embedding layer
        self.card_embedding = nn.Embedding(
            num_card_types + 1, card_embedding_dim  # +1 for padding token
        )

        # Card branch (3 layers)
        self.card_branch = nn.Sequential(
            nn.Linear(card_embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim // 2),
        )

        # History branch (2 layers)
        self.history_branch = nn.Sequential(
            nn.Linear(history_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim // 2),
        )

        # Combined layers (3 layers total with skip connection)
        combined_dim = hidden_dim  # Both branches output hidden_dim // 2

        self.combined_layers = nn.ModuleList(
            [
                nn.Linear(combined_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            ]
        )

        self.combined_norms = nn.ModuleList(
            [
                nn.BatchNorm1d(hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            ]
        )

        # Skip connection projection
        self.skip_projection = nn.Linear(combined_dim, hidden_dim)

        # Output layer
        self.output = nn.Linear(hidden_dim, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(
        self,
        card_indices: torch.Tensor,  # [batch, num_cards]
        history_features: torch.Tensor,  # [batch, history_dim]
    ) -> torch.Tensor:
        """Forward pass through the network."""
        # Process cards
        card_embeds = self.card_embedding(card_indices)  # [batch, num_cards, embed_dim]
        card_features = card_embeds.mean(dim=1)  # Average pooling
        card_out = self.card_branch(card_features)  # [batch, hidden_dim // 2]

        # Process history
        history_out = self.history_branch(history_features)  # [batch, hidden_dim // 2]

        # Combine branches
        combined = torch.cat([card_out, history_out], dim=-1)  # [batch, hidden_dim]

        # Combined layers with skip connection
        x = F.relu(self.combined_norms[0](self.combined_layers[0](combined)))
        x = self.dropout(x)

        # Second layer with skip
        residual = self.skip_projection(combined)
        x = F.relu(self.combined_norms[1](self.combined_layers[1](x) + residual))
        x = self.dropout(x)

        # Output single value
        return self.output(x)


class StateEncoder:
    """Encodes game states and actions for neural network input."""

    def __init__(self, num_card_types: int = 32, max_cards_encoded: int = 30):
        self.num_card_types = num_card_types
        self.max_cards_encoded = max_cards_encoded
        self.history_dim = 64

        # Build card name to index mapping
        from ..core import CardDatabase

        CardDatabase.initialize()
        cards = CardDatabase.get_all_cards()

        self.card_to_idx = {}
        for i, name in enumerate(sorted(cards.keys())):
            self.card_to_idx[name] = i + 1  # Reserve 0 for padding

    def encode_state_and_action(
        self, state: GameState, player: Player, action: Action
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a state-action pair for the network."""
        # Encode cards
        card_indices = self._encode_cards(state, player, action)

        # Encode history features
        history_features = self._encode_history(state, player, action)

        return card_indices, history_features

    def _encode_cards(self, state: GameState, player: Player, action: Action) -> torch.Tensor:
        """Encode cards as indices."""
        indices = torch.zeros(self.max_cards_encoded, dtype=torch.long)
        idx = 0

        # Cards in hand
        for card in state.hands[player]:
            if idx < self.max_cards_encoded:
                indices[idx] = self.card_to_idx.get(card.name, 0)
                idx += 1

        # Player's creatures
        for creature in state.play_areas[player]:
            if idx < self.max_cards_encoded:
                indices[idx] = self.card_to_idx.get(creature.card.name, 0)
                idx += 1

        # Opponent's creatures
        for creature in state.play_areas[player.other()]:
            if idx < self.max_cards_encoded:
                indices[idx] = self.card_to_idx.get(creature.card.name, 0)
                idx += 1

        # Action card if playing
        if action.card and idx < self.max_cards_encoded:
            indices[idx] = self.card_to_idx.get(action.card.name, 0)

        return indices

    def _encode_history(self, state: GameState, player: Player, action: Action) -> torch.Tensor:
        """Encode game state as numerical features."""
        features = torch.zeros(self.history_dim, dtype=torch.float32)

        opponent = player.other()

        # Core game state (0-15)
        features[0] = state.life[player] / 10.0
        features[1] = state.life[opponent] / 10.0
        features[2] = state.mindbugs_available[player] / 2.0
        features[3] = state.mindbugs_available[opponent] / 2.0
        features[4] = float(state.current_player == player)
        features[5] = float(state.extra_turn_pending == player)
        features[6] = float(state.mindbug_decision_pending)
        features[7] = float(state.hunter_choice_pending)
        features[8] = len(state.hands[player]) / 5.0
        features[9] = len(state.hands[opponent]) / 5.0
        features[10] = len(state.play_areas[player]) / 5.0
        features[11] = len(state.play_areas[opponent]) / 5.0
        features[12] = len(state.decks[player]) / 10.0
        features[13] = len(state.decks[opponent]) / 10.0
        features[14] = len(state.discard_piles[player]) / 20.0
        features[15] = len(state.discard_piles[opponent]) / 20.0

        # Action encoding (16-22)
        action_offset = 16
        action_type_idx = list(ActionType).index(action.action_type)
        if action_type_idx < 6:
            features[action_offset + action_type_idx] = 1.0

        features[21] = action.creature_index / 10.0 if action.creature_index is not None else 0.0
        features[22] = action.target_index / 10.0 if action.target_index is not None else 0.0

        # Combat state (23-24)
        if state.attacking_creature:
            features[23] = 1.0
            features[24] = state.attacking_creature.card.power / 10.0

        # Creature power statistics (25-30)
        my_powers = [
            c.get_effective_power(state.current_player == player, state.play_areas[player])
            for c in state.play_areas[player]
        ]
        opp_powers = [
            c.get_effective_power(state.current_player == opponent, state.play_areas[opponent])
            for c in state.play_areas[opponent]
        ]

        if my_powers:
            features[25] = min(my_powers) / 10.0
            features[26] = max(my_powers) / 10.0
            features[27] = sum(my_powers) / len(my_powers) / 10.0

        if opp_powers:
            features[28] = min(opp_powers) / 10.0
            features[29] = max(opp_powers) / 10.0
            features[30] = sum(opp_powers) / len(opp_powers) / 10.0

        # Keyword presence (31-40)
        from ..core import Keyword

        keyword_offset = 31
        my_keywords = set()
        opp_keywords = set()

        for c in state.play_areas[player]:
            my_keywords.update(
                c.get_effective_keywords(state.play_areas[player], state.play_areas[opponent])
            )

        for c in state.play_areas[opponent]:
            opp_keywords.update(
                c.get_effective_keywords(state.play_areas[opponent], state.play_areas[player])
            )

        for i, keyword in enumerate(
            [Keyword.POISONOUS, Keyword.SNEAKY, Keyword.HUNTER, Keyword.TOUGH, Keyword.FRENZY]
        ):
            features[keyword_offset + i * 2] = float(keyword in my_keywords)
            features[keyword_offset + i * 2 + 1] = float(keyword in opp_keywords)

        # Special effects (41-44)
        features[41] = float(state.deathweaver_active[player])
        features[42] = float(state.deathweaver_active[opponent])
        features[43] = float(any(c.card.name == "Elephantopus" for c in state.play_areas[player]))
        features[44] = float(any(c.card.name == "Elephantopus" for c in state.play_areas[opponent]))

        # Action availability (45-50)
        features[45] = float(len(state.hands[player]) > 0)
        features[46] = float(
            len(
                [
                    c
                    for c in state.play_areas[player]
                    if id(c) not in state.creatures_attacked_this_turn
                ]
            )
            > 0
        )
        features[47] = float(state.mindbugs_available[player] > 0)
        features[48] = float(state.mindbugs_available[opponent] > 0)
        features[49] = float(len(state.valid_blockers) > 0 if state.hunter_choice_pending else 0)
        features[50] = float(len(state.play_areas[opponent]) == 0)

        # Turn and phase (51-55)
        features[51] = float(state.current_player == player)
        features[52] = float(state.mindbug_decision_pending)
        features[53] = float(state.hunter_choice_pending)
        features[54] = float(state.extra_turn_pending is not None)
        features[55] = float(len(state.creatures_attacked_this_turn) > 0)

        # Resource depletion (56-61)
        features[56] = float(len(state.decks[player]) == 0)
        features[57] = float(len(state.decks[opponent]) == 0)
        features[58] = float(len(state.hands[player]) == 0)
        features[59] = float(len(state.hands[opponent]) == 0)
        features[60] = float(state.life[player] == 1)
        features[61] = float(state.life[opponent] == 1)

        # Unique creatures count (62-63)
        features[62] = len(set(c.card.name for c in state.play_areas[player])) / 10.0
        features[63] = len(set(c.card.name for c in state.play_areas[opponent])) / 10.0

        return features

    def batch_encode(
        self, states: List[GameState], players: List[Player], actions: List[Action]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Efficiently encode multiple state-action pairs."""
        batch_size = len(states)

        card_indices = torch.zeros((batch_size, self.max_cards_encoded), dtype=torch.long)
        history_features = torch.zeros((batch_size, self.history_dim), dtype=torch.float32)

        for i, (state, player, action) in enumerate(zip(states, players, actions)):
            cards, history = self.encode_state_and_action(state, player, action)
            card_indices[i] = cards
            history_features[i] = history

        return card_indices, history_features
