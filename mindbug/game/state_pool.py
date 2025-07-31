# Optimized state management with object pooling and fast copying
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from dataclasses import dataclass, field

from .constants import Player, STARTING_LIFE, MINDBUGS_PER_PLAYER
from .cards import Card
from .state import CreatureState, GameState


class StatePool:
    """
    Object pool for efficient game state management.
    Avoids expensive deepcopy operations during MCTS traversals.
    """
    
    def __init__(self, pool_size: int = 10000):
        self.pool_size = pool_size
        self.states: List[GameState] = []
        self.available: List[GameState] = []
        
        # Pre-allocate states
        for _ in range(pool_size):
            state = GameState(current_player=Player.PLAYER_1)
            self.states.append(state)
            self.available.append(state)
    
    def acquire(self) -> GameState:
        """Get a state from the pool."""
        if self.available:
            return self.available.pop()
        else:
            # Pool exhausted, create new state
            return GameState(current_player=Player.PLAYER_1)
    
    def release(self, state: GameState):
        """Return a state to the pool."""
        if len(self.available) < self.pool_size:
            self._reset_state(state)
            self.available.append(state)
    
    def _reset_state(self, state: GameState):
        """Reset state to initial conditions."""
        state.current_player = Player.PLAYER_1
        state.life = {Player.PLAYER_1: STARTING_LIFE, Player.PLAYER_2: STARTING_LIFE}
        state.mindbugs_available = {Player.PLAYER_1: MINDBUGS_PER_PLAYER, Player.PLAYER_2: MINDBUGS_PER_PLAYER}
        
        for player in [Player.PLAYER_1, Player.PLAYER_2]:
            state.hands[player].clear()
            state.decks[player].clear()
            state.play_areas[player].clear()
            state.discard_piles[player].clear()
        
        state.extra_turn_pending = None
        state.mindbug_decision_pending = False
        state.pending_creature_play = None
        state.hunter_choice_pending = False
        state.valid_blockers.clear()
        state.attacking_creature = None
        state.deathweaver_active = {Player.PLAYER_1: False, Player.PLAYER_2: False}
        state.creatures_attacked_this_turn.clear()


class FastGameState:
    """
    Optimized game state using numpy arrays for faster copying.
    Designed for high-performance Deep CFR training.
    """
    
    # Constants for array indices
    LIFE_P1 = 0
    LIFE_P2 = 1
    MINDBUGS_P1 = 2
    MINDBUGS_P2 = 3
    CURRENT_PLAYER = 4
    EXTRA_TURN = 5  # 0=none, 1=P1, 2=P2
    MINDBUG_PENDING = 6
    HUNTER_PENDING = 7
    DEATHWEAVER_P1 = 8
    DEATHWEAVER_P2 = 9
    
    MAX_HAND = 10
    MAX_CREATURES = 10
    MAX_DECK = 10
    MAX_DISCARD = 48
    
    def __init__(self):
        # Core state as numpy array for fast copying
        self.core_state = np.zeros(10, dtype=np.int8)
        self.core_state[self.LIFE_P1] = STARTING_LIFE
        self.core_state[self.LIFE_P2] = STARTING_LIFE
        self.core_state[self.MINDBUGS_P1] = MINDBUGS_PER_PLAYER
        self.core_state[self.MINDBUGS_P2] = MINDBUGS_PER_PLAYER
        
        # Card zones as fixed-size arrays (-1 = empty slot)
        self.hands = {
            Player.PLAYER_1: np.full(self.MAX_HAND, -1, dtype=np.int8),
            Player.PLAYER_2: np.full(self.MAX_HAND, -1, dtype=np.int8)
        }
        self.hand_sizes = {Player.PLAYER_1: 0, Player.PLAYER_2: 0}
        
        self.decks = {
            Player.PLAYER_1: np.full(self.MAX_DECK, -1, dtype=np.int8),
            Player.PLAYER_2: np.full(self.MAX_DECK, -1, dtype=np.int8)
        }
        self.deck_sizes = {Player.PLAYER_1: 0, Player.PLAYER_2: 0}
        
        self.discard = {
            Player.PLAYER_1: np.full(self.MAX_DISCARD, -1, dtype=np.int8),
            Player.PLAYER_2: np.full(self.MAX_DISCARD, -1, dtype=np.int8)
        }
        self.discard_sizes = {Player.PLAYER_1: 0, Player.PLAYER_2: 0}
        
        # Creatures as structured arrays
        self.creatures = {
            Player.PLAYER_1: np.zeros(self.MAX_CREATURES, dtype=[
                ('card_id', 'i1'),
                ('controller', 'i1'),
                ('owner', 'i1'),
                ('exhausted', 'i1'),
                ('attack_count', 'i1'),
                ('active', 'i1')
            ]),
            Player.PLAYER_2: np.zeros(self.MAX_CREATURES, dtype=[
                ('card_id', 'i1'),
                ('controller', 'i1'),
                ('owner', 'i1'),
                ('exhausted', 'i1'),
                ('attack_count', 'i1'),
                ('active', 'i1')
            ])
        }
        self.creature_counts = {Player.PLAYER_1: 0, Player.PLAYER_2: 0}
        
        # Combat state
        self.attacking_creature_idx = -1
        self.attacking_player = 0
        self.valid_blockers = np.full(self.MAX_CREATURES, -1, dtype=np.int8)
        self.valid_blocker_count = 0
        
        # Pending play state
        self.pending_player = 0
        self.pending_card = -1
        
        # Attack tracking
        self.creatures_attacked = np.zeros(self.MAX_CREATURES * 2, dtype=np.int8)
        
        # Card ID mapping (shared across all states)
        self._card_to_id = {}
        self._id_to_card = {}
        self._init_card_mapping()
    
    def _init_card_mapping(self):
        """Initialize card to ID mapping."""
        from .cards import CardDefinitions
        cards = CardDefinitions.get_first_contact_cards()
        for i, (name, card) in enumerate(sorted(cards.items())):
            self._card_to_id[name] = i
            self._id_to_card[i] = card
    
    def copy(self) -> 'FastGameState':
        """Fast copy using numpy array operations."""
        new_state = FastGameState()
        
        # Copy core state
        new_state.core_state = self.core_state.copy()
        
        # Copy card zones
        for player in [Player.PLAYER_1, Player.PLAYER_2]:
            new_state.hands[player] = self.hands[player].copy()
            new_state.hand_sizes[player] = self.hand_sizes[player]
            
            new_state.decks[player] = self.decks[player].copy()
            new_state.deck_sizes[player] = self.deck_sizes[player]
            
            new_state.discard[player] = self.discard[player].copy()
            new_state.discard_sizes[player] = self.discard_sizes[player]
            
            new_state.creatures[player] = self.creatures[player].copy()
            new_state.creature_counts[player] = self.creature_counts[player]
        
        # Copy combat state
        new_state.attacking_creature_idx = self.attacking_creature_idx
        new_state.attacking_player = self.attacking_player
        new_state.valid_blockers = self.valid_blockers.copy()
        new_state.valid_blocker_count = self.valid_blocker_count
        
        # Copy pending state
        new_state.pending_player = self.pending_player
        new_state.pending_card = self.pending_card
        
        # Copy attack tracking
        new_state.creatures_attacked = self.creatures_attacked.copy()
        
        # Share card mappings (no need to copy)
        new_state._card_to_id = self._card_to_id
        new_state._id_to_card = self._id_to_card
        
        return new_state
    
    def to_game_state(self) -> GameState:
        """Convert to standard GameState for compatibility."""
        state = GameState(
            current_player=Player.PLAYER_1 if self.core_state[self.CURRENT_PLAYER] == 0 else Player.PLAYER_2
        )
        
        state.life[Player.PLAYER_1] = int(self.core_state[self.LIFE_P1])
        state.life[Player.PLAYER_2] = int(self.core_state[self.LIFE_P2])
        state.mindbugs_available[Player.PLAYER_1] = int(self.core_state[self.MINDBUGS_P1])
        state.mindbugs_available[Player.PLAYER_2] = int(self.core_state[self.MINDBUGS_P2])
        
        # Convert card zones
        for player in [Player.PLAYER_1, Player.PLAYER_2]:
            # Hands
            for i in range(self.hand_sizes[player]):
                card_id = self.hands[player][i]
                if card_id >= 0:
                    state.hands[player].append(self._id_to_card[card_id])
            
            # Decks
            for i in range(self.deck_sizes[player]):
                card_id = self.decks[player][i]
                if card_id >= 0:
                    state.decks[player].append(self._id_to_card[card_id])
            
            # Discard
            for i in range(self.discard_sizes[player]):
                card_id = self.discard[player][i]
                if card_id >= 0:
                    state.discard_piles[player].append(self._id_to_card[card_id])
            
            # Creatures
            for i in range(self.creature_counts[player]):
                c = self.creatures[player][i]
                if c['active']:
                    creature = CreatureState(
                        card=self._id_to_card[c['card_id']],
                        controller=Player.PLAYER_1 if c['controller'] == 0 else Player.PLAYER_2,
                        owner=Player.PLAYER_1 if c['owner'] == 0 else Player.PLAYER_2,
                        is_exhausted=bool(c['exhausted']),
                        attack_count=int(c['attack_count'])
                    )
                    state.play_areas[player].append(creature)
        
        # Convert state flags
        state.extra_turn_pending = None
        if self.core_state[self.EXTRA_TURN] == 1:
            state.extra_turn_pending = Player.PLAYER_1
        elif self.core_state[self.EXTRA_TURN] == 2:
            state.extra_turn_pending = Player.PLAYER_2
        
        state.mindbug_decision_pending = bool(self.core_state[self.MINDBUG_PENDING])
        state.hunter_choice_pending = bool(self.core_state[self.HUNTER_PENDING])
        state.deathweaver_active[Player.PLAYER_1] = bool(self.core_state[self.DEATHWEAVER_P1])
        state.deathweaver_active[Player.PLAYER_2] = bool(self.core_state[self.DEATHWEAVER_P2])
        
        # Convert pending play
        if state.mindbug_decision_pending and self.pending_card >= 0:
            pending_player = Player.PLAYER_1 if self.pending_player == 0 else Player.PLAYER_2
            state.pending_creature_play = (pending_player, self._id_to_card[self.pending_card])
        
        # Convert valid blockers
        for i in range(self.valid_blocker_count):
            if self.valid_blockers[i] >= 0:
                state.valid_blockers.append(int(self.valid_blockers[i]))
        
        # Convert attack tracking
        for i in range(len(self.creatures_attacked)):
            if self.creatures_attacked[i]:
                # Map back to creature IDs
                player_idx = i // self.MAX_CREATURES
                creature_idx = i % self.MAX_CREATURES
                player = Player.PLAYER_1 if player_idx == 0 else Player.PLAYER_2
                if creature_idx < len(state.play_areas[player]):
                    state.creatures_attacked_this_turn.add(id(state.play_areas[player][creature_idx]))
        
        state.update_creature_indices()
        return state
    
    @staticmethod
    def from_game_state(state: GameState) -> 'FastGameState':
        """Create FastGameState from standard GameState."""
        fast = FastGameState()
        
        # Core state
        fast.core_state[fast.LIFE_P1] = state.life[Player.PLAYER_1]
        fast.core_state[fast.LIFE_P2] = state.life[Player.PLAYER_2]
        fast.core_state[fast.MINDBUGS_P1] = state.mindbugs_available[Player.PLAYER_1]
        fast.core_state[fast.MINDBUGS_P2] = state.mindbugs_available[Player.PLAYER_2]
        fast.core_state[fast.CURRENT_PLAYER] = 0 if state.current_player == Player.PLAYER_1 else 1
        
        if state.extra_turn_pending == Player.PLAYER_1:
            fast.core_state[fast.EXTRA_TURN] = 1
        elif state.extra_turn_pending == Player.PLAYER_2:
            fast.core_state[fast.EXTRA_TURN] = 2
        
        fast.core_state[fast.MINDBUG_PENDING] = int(state.mindbug_decision_pending)
        fast.core_state[fast.HUNTER_PENDING] = int(state.hunter_choice_pending)
        fast.core_state[fast.DEATHWEAVER_P1] = int(state.deathweaver_active[Player.PLAYER_1])
        fast.core_state[fast.DEATHWEAVER_P2] = int(state.deathweaver_active[Player.PLAYER_2])
        
        # Convert zones
        for player in [Player.PLAYER_1, Player.PLAYER_2]:
            # Hands
            for i, card in enumerate(state.hands[player][:fast.MAX_HAND]):
                fast.hands[player][i] = fast._card_to_id.get(card.name, -1)
            fast.hand_sizes[player] = len(state.hands[player])
            
            # Decks
            for i, card in enumerate(state.decks[player][:fast.MAX_DECK]):
                fast.decks[player][i] = fast._card_to_id.get(card.name, -1)
            fast.deck_sizes[player] = len(state.decks[player])
            
            # Discard
            for i, card in enumerate(state.discard_piles[player][:fast.MAX_DISCARD]):
                fast.discard[player][i] = fast._card_to_id.get(card.name, -1)
            fast.discard_sizes[player] = len(state.discard_piles[player])
            
            # Creatures
            for i, creature in enumerate(state.play_areas[player][:fast.MAX_CREATURES]):
                fast.creatures[player][i]['card_id'] = fast._card_to_id.get(creature.card.name, -1)
                fast.creatures[player][i]['controller'] = 0 if creature.controller == Player.PLAYER_1 else 1
                fast.creatures[player][i]['owner'] = 0 if creature.owner == Player.PLAYER_1 else 1
                fast.creatures[player][i]['exhausted'] = int(creature.is_exhausted)
                fast.creatures[player][i]['attack_count'] = creature.attack_count
                fast.creatures[player][i]['active'] = 1
            fast.creature_counts[player] = len(state.play_areas[player])
        
        # Pending play
        if state.pending_creature_play:
            player, card = state.pending_creature_play
            fast.pending_player = 0 if player == Player.PLAYER_1 else 1
            fast.pending_card = fast._card_to_id.get(card.name, -1)
        
        # Valid blockers
        for i, idx in enumerate(state.valid_blockers[:fast.MAX_CREATURES]):
            fast.valid_blockers[i] = idx
        fast.valid_blocker_count = len(state.valid_blockers)
        
        return fast