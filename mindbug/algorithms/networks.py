from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualBranchMindbugNetwork(nn.Module):
    # 7-layer dual-branch architecture as specified in Deep CFR paper
    # Card branch: 3 layers, History branch: 2 layers, Combined: 3 layers
    def __init__(self, card_embedding_dim: int = 128, history_dim: int = 64, 
                 hidden_dim: int = 256, num_card_types: int = 32):
        super().__init__()
        
        # Card embedding layer - learns representations for each card type
        self.card_embedding = nn.Embedding(num_card_types + 1, card_embedding_dim)  # +1 for padding
        
        # Card branch processes card-specific information
        self.card_branch = nn.Sequential(
            nn.Linear(card_embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim // 2)
        )
        
        # History branch processes numerical game state features
        self.history_branch = nn.Sequential(
            nn.Linear(history_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim // 2)
        )
        
        # Combined layers process merged features
        combined_dim = hidden_dim  # Both branches output hidden_dim // 2
        
        self.combined_fc1 = nn.Linear(combined_dim, hidden_dim)
        self.combined_bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.combined_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.combined_bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Skip connection for better gradient flow
        self.skip_projection = nn.Linear(combined_dim, hidden_dim)
        
        # Final output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # Regularization
        self.dropout = nn.Dropout(0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        # Xavier initialization for better convergence
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
    
    def forward(self, card_indices: torch.Tensor, history_features: torch.Tensor) -> torch.Tensor:
        # Process card indices through embedding and card branch
        card_embed = self.card_embedding(card_indices)  # [batch, max_cards, embed_dim]
        
        # Average pooling over card sequence
        card_embed = card_embed.mean(dim=1)  # [batch, embed_dim]
        card_features = self.card_branch(card_embed)  # [batch, hidden_dim // 2]
        
        # Process history features through history branch
        history_features = self.history_branch(history_features)  # [batch, hidden_dim // 2]
        
        # Combine both branches
        combined = torch.cat([card_features, history_features], dim=-1)  # [batch, hidden_dim]
        
        # First combined layer with dropout
        x = F.relu(self.combined_bn1(self.combined_fc1(combined)), inplace=True)
        x = self.dropout(x)
        
        # Second combined layer with skip connection
        residual = self.skip_projection(combined)
        x = F.relu(self.combined_bn2(self.combined_fc2(x) + residual), inplace=True)
        x = self.dropout(x)
        
        # Output single value (advantage or strategy)
        return self.output_layer(x)


class MindbugStateEncoder:
    # Encodes game states and actions for neural network input
    def __init__(self, num_card_types: int = 32, max_cards_encoded: int = 30):
        self.num_card_types = num_card_types
        self.max_cards_encoded = max_cards_encoded
        self.card_name_to_idx = {}
        self._initialize_card_indices()
        
        # Fixed history feature dimension
        self.history_dim = 64
    
    def _initialize_card_indices(self):
        # Create consistent card name to index mapping
        from ..game.cards import CardDefinitions
        cards = CardDefinitions.get_first_contact_cards()
        
        # Sort for deterministic ordering
        for i, name in enumerate(sorted(cards.keys())):
            self.card_name_to_idx[name] = i + 1  # Reserve 0 for padding
    
    def encode_state_and_action(
        self, state, player, action
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode cards as indices
        card_indices = torch.zeros(self.max_cards_encoded, dtype=torch.long)
        idx = 0
        
        # Cards in hand
        for card in state.hands[player]:
            if idx < self.max_cards_encoded:
                card_indices[idx] = self._get_card_index(card)
                idx += 1
        
        # Player's creatures
        for creature in state.play_areas[player]:
            if idx < self.max_cards_encoded:
                card_indices[idx] = self._get_card_index(creature.card)
                idx += 1
        
        # Opponent's creatures
        for creature in state.play_areas[player.other()]:
            if idx < self.max_cards_encoded:
                card_indices[idx] = self._get_card_index(creature.card)
                idx += 1
        
        # Action card if playing a creature
        if action.card and idx < self.max_cards_encoded:
            card_indices[idx] = self._get_card_index(action.card)
        
        # Build history features
        history_features = self._build_history_features(state, player, action)
        
        return card_indices, history_features
    
    def _build_history_features(self, state, player, action) -> torch.Tensor:
        features = torch.zeros(self.history_dim, dtype=torch.float32)
        
        # Core game state (indices 0-15)
        features[0] = state.life[player] / 10.0
        features[1] = state.life[player.other()] / 10.0
        features[2] = state.mindbugs_available[player] / 2.0
        features[3] = state.mindbugs_available[player.other()] / 2.0
        features[4] = float(state.current_player == player)
        features[5] = float(state.extra_turn_pending == player)
        features[6] = float(state.mindbug_decision_pending)
        features[7] = float(state.hunter_choice_pending)
        features[8] = len(state.hands[player]) / 5.0
        features[9] = len(state.hands[player.other()]) / 5.0
        features[10] = len(state.play_areas[player]) / 5.0
        features[11] = len(state.play_areas[player.other()]) / 5.0
        features[12] = len(state.decks[player]) / 10.0
        features[13] = len(state.decks[player.other()]) / 10.0
        features[14] = len(state.discard_piles[player]) / 20.0
        features[15] = len(state.discard_piles[player.other()]) / 20.0
        
        # Action encoding (indices 16-22)
        action_offset = 16
        if action.action_type.value <= 5:  # Safety check
            features[action_offset + action.action_type.value - 1] = 1.0
        
        features[21] = action.creature_index / 10.0 if action.creature_index is not None else 0.0
        features[22] = action.target_index / 10.0 if action.target_index is not None else 0.0
        
        # Combat state (indices 23-24)
        if state.attacking_creature:
            features[23] = 1.0
            features[24] = state.attacking_creature.card.power / 10.0
        
        # Creature power statistics (indices 25-30)
        my_powers = [c.get_effective_power(state.current_player == player, state.play_areas[player]) 
                     for c in state.play_areas[player]]
        opp_powers = [c.get_effective_power(state.current_player == player.other(), state.play_areas[player.other()]) 
                      for c in state.play_areas[player.other()]]
        
        if my_powers:
            features[25] = min(my_powers) / 10.0
            features[26] = max(my_powers) / 10.0
            features[27] = sum(my_powers) / len(my_powers) / 10.0
        
        if opp_powers:
            features[28] = min(opp_powers) / 10.0
            features[29] = max(opp_powers) / 10.0
            features[30] = sum(opp_powers) / len(opp_powers) / 10.0
        
        # Keyword presence flags (indices 31-40)
        keyword_offset = 31
        my_keywords = set()
        opp_keywords = set()
        
        for c in state.play_areas[player]:
            my_keywords.update(c.get_effective_keywords(state.play_areas[player], state.play_areas[player.other()]))
        for c in state.play_areas[player.other()]:
            opp_keywords.update(c.get_effective_keywords(state.play_areas[player.other()], state.play_areas[player]))
        
        from ..game.constants import Keyword
        for i, keyword in enumerate([Keyword.POISONOUS, Keyword.SNEAKY, Keyword.HUNTER, Keyword.TOUGH, Keyword.FRENZY]):
            features[keyword_offset + i * 2] = float(keyword in my_keywords)
            features[keyword_offset + i * 2 + 1] = float(keyword in opp_keywords)
        
        # Special effects active (indices 41-44)
        features[41] = float(state.deathweaver_active[player])
        features[42] = float(state.deathweaver_active[player.other()])
        features[43] = float(any(c.card.name == "Elephantopus" for c in state.play_areas[player]))
        features[44] = float(any(c.card.name == "Elephantopus" for c in state.play_areas[player.other()]))
        
        # Action availability (indices 45-50)
        features[45] = float(len(state.hands[player]) > 0)
        features[46] = float(len([c for c in state.play_areas[player] if id(c) not in state.creatures_attacked_this_turn]) > 0)
        features[47] = float(state.mindbugs_available[player] > 0)
        features[48] = float(state.mindbugs_available[player.other()] > 0)
        features[49] = float(len(state.valid_blockers) > 0 if state.hunter_choice_pending else 0)
        features[50] = float(len(state.play_areas[player.other()]) == 0)  # Can attack directly
        
        # Turn and phase indicators (indices 51-55)
        features[51] = float(state.current_player == player)
        features[52] = float(state.mindbug_decision_pending)
        features[53] = float(state.hunter_choice_pending)
        features[54] = float(state.extra_turn_pending is not None)
        features[55] = float(len(state.creatures_attacked_this_turn) > 0)
        
        # Resource depletion indicators (indices 56-61)
        features[56] = float(len(state.decks[player]) == 0)
        features[57] = float(len(state.decks[player.other()]) == 0)
        features[58] = float(len(state.hands[player]) == 0)
        features[59] = float(len(state.hands[player.other()]) == 0)
        features[60] = float(state.life[player] == 1)
        features[61] = float(state.life[player.other()] == 1)
        
        # Indices 62-63 reserved for future use
        
        return features
    
    def _get_card_index(self, card) -> int:
        # Map card name to index, return 0 if unknown
        return self.card_name_to_idx.get(card.name, 0)
    
    def batch_encode(
        self, 
        states: List, 
        players: List, 
        actions: List
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Efficiently encode multiple state-action pairs
        batch_size = len(states)
        
        # Pre-allocate tensors for entire batch
        card_indices_batch = torch.zeros((batch_size, self.max_cards_encoded), dtype=torch.long)
        history_features_batch = torch.zeros((batch_size, self.history_dim), dtype=torch.float32)
        
        # Encode each sample
        for i, (state, player, action) in enumerate(zip(states, players, actions)):
            card_indices, history_features = self.encode_state_and_action(state, player, action)
            card_indices_batch[i] = card_indices
            history_features_batch[i] = history_features
        
        return card_indices_batch, history_features_batch