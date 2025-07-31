import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .actions import Action
from .cards import Card
from .constants import (
    MAX_HAND_SIZE,
    MINDBUGS_PER_PLAYER,
    STARTING_HAND_SIZE,
    STARTING_LIFE,
    ActionType,
    Keyword,
    Player,
    Zone,
)


@dataclass
class CreatureState:
    # Represents a creature in play with its current state
    card: Card
    controller: Player  # Who currently controls it
    owner: Player      # Who originally played it (for discard)
    is_exhausted: bool = False  # TOUGH keyword exhaustion
    attack_count: int = 0       # Tracks attacks for FRENZY
    play_area_index: Optional[int] = None

    def get_effective_power(
        self, is_controllers_turn: bool, allied_creatures: List["CreatureState"]
    ) -> int:
        # Calculate power including all modifiers
        power = self.card.power
        
        # Goblin Werewolf: +6 on controller's turn
        if self.card.name == "Goblin Werewolf" and is_controllers_turn:
            power += 6
        
        # Lone Yeti: +5 when alone
        if self.card.name == "Lone Yeti" and len(allied_creatures) == 1:
            power += 5
        
        # Shield Bugs buff other creatures
        for creature in allied_creatures:
            if creature.card.name == "Shield Bugs" and creature != self:
                power += 1
        
        # Urchin Hurler buff on controller's turn
        if is_controllers_turn:
            for creature in allied_creatures:
                if creature.card.name == "Urchin Hurler" and creature != self:
                    power += 2
        
        return max(1, power)  # Minimum 1 power

    def get_effective_keywords(
        self,
        allied_creatures: List["CreatureState"],
        enemy_creatures: List["CreatureState"],
    ) -> Set[Keyword]:
        # Get all keywords including dynamic ones
        keywords = self.card.keywords.copy()
        
        # Lone Yeti gains FRENZY when alone
        if self.card.name == "Lone Yeti" and len(allied_creatures) == 1:
            keywords.add(Keyword.FRENZY)
        
        # Sharky Crab-Dog-Mummypus copies enemy keywords
        if self.card.name == "Sharky Crab-Dog-Mummypus":
            enemy_keywords = set()
            for creature in enemy_creatures:
                # Get base keywords
                enemy_effective = creature.card.keywords.copy()
                
                # Check for Lone Yeti FRENZY
                if creature.card.name == "Lone Yeti" and len(enemy_creatures) == 1:
                    enemy_effective.add(Keyword.FRENZY)
                
                # Check for Snail Thrower effects
                for ally in enemy_creatures:
                    if ally.card.name == "Snail Thrower" and ally != creature and creature.card.power <= 4:
                        enemy_effective.add(Keyword.HUNTER)
                        enemy_effective.add(Keyword.POISONOUS)
                
                enemy_keywords.update(enemy_effective)
            
            # Copy relevant keywords
            copyable = {Keyword.HUNTER, Keyword.SNEAKY, Keyword.FRENZY, Keyword.POISONOUS}
            keywords.update(enemy_keywords & copyable)
        
        # Snail Thrower grants keywords to small allies
        for creature in allied_creatures:
            if (
                creature.card.name == "Snail Thrower"
                and creature != self
                and self.card.power <= 4
            ):
                keywords.add(Keyword.HUNTER)
                keywords.add(Keyword.POISONOUS)
        
        return keywords


@dataclass
class GameState:
    # Complete game state representation
    current_player: Player
    
    # Core resources
    life: Dict[Player, int] = field(
        default_factory=lambda: {
            Player.PLAYER_1: STARTING_LIFE,
            Player.PLAYER_2: STARTING_LIFE,
        }
    )
    mindbugs_available: Dict[Player, int] = field(
        default_factory=lambda: {
            Player.PLAYER_1: MINDBUGS_PER_PLAYER,
            Player.PLAYER_2: MINDBUGS_PER_PLAYER,
        }
    )
    
    # Card zones
    hands: Dict[Player, List[Card]] = field(
        default_factory=lambda: {Player.PLAYER_1: [], Player.PLAYER_2: []}
    )
    decks: Dict[Player, List[Card]] = field(
        default_factory=lambda: {Player.PLAYER_1: [], Player.PLAYER_2: []}
    )
    play_areas: Dict[Player, List[CreatureState]] = field(
        default_factory=lambda: {Player.PLAYER_1: [], Player.PLAYER_2: []}
    )
    discard_piles: Dict[Player, List[Card]] = field(
        default_factory=lambda: {Player.PLAYER_1: [], Player.PLAYER_2: []}
    )
    
    # Turn state
    extra_turn_pending: Optional[Player] = None
    
    # Mindbug decision state
    mindbug_decision_pending: bool = False
    pending_creature_play: Optional[Tuple[Player, Card]] = None
    
    # Combat state
    hunter_choice_pending: bool = False
    valid_blockers: List[int] = field(default_factory=list)
    attacking_creature: Optional[CreatureState] = None
    
    # Effect state
    deathweaver_active: Dict[Player, bool] = field(
        default_factory=lambda: {Player.PLAYER_1: False, Player.PLAYER_2: False}
    )
    
    # Attack tracking for FRENZY
    creatures_attacked_this_turn: Set[int] = field(default_factory=set)

    def copy(self) -> "GameState":
        # Create deep copy of state
        # TODO: Optimize with state pool for performance
        return deepcopy(self)

    def is_terminal(self) -> bool:
        # Check if game has ended
        # Life loss
        if any(life <= 0 for life in self.life.values()):
            return True
        
        # No legal actions (rare)
        legal_actions = self.get_legal_actions()
        if (
            not legal_actions
            and not self.mindbug_decision_pending
            and not self.hunter_choice_pending
        ):
            return True
        
        return False

    def get_winner(self) -> Optional[Player]:
        # Determine winner if game is terminal
        if not self.is_terminal():
            return None
        
        # Check life totals
        if self.life[Player.PLAYER_1] <= 0 and self.life[Player.PLAYER_2] <= 0:
            return None  # Draw
        elif self.life[Player.PLAYER_1] <= 0:
            return Player.PLAYER_2
        elif self.life[Player.PLAYER_2] <= 0:
            return Player.PLAYER_1
        
        # Terminal due to no actions - opponent wins
        return self.current_player.other()

    def get_legal_actions(self) -> List[Action]:
        # Get all legal actions for current game state
        actions = []
        
        # Mindbug decision phase
        if self.mindbug_decision_pending and self.pending_creature_play:
            opponent = self.current_player
            if self.mindbugs_available[opponent] > 0:
                actions.append(Action(ActionType.USE_MINDBUG, opponent))
            actions.append(Action(ActionType.PASS_MINDBUG, opponent))
            return actions
        
        # Hunter blocker choice
        if self.hunter_choice_pending and self.valid_blockers:
            for blocker_idx in self.valid_blockers:
                actions.append(
                    Action(
                        ActionType.CHOOSE_BLOCKER,
                        self.current_player,
                        target_index=blocker_idx,
                    )
                )
            return actions
        
        # Normal turn - play creatures
        for card in self.hands[self.current_player]:
            actions.append(
                Action(ActionType.PLAY_CREATURE, self.current_player, card=card)
            )
        
        # Normal turn - attack with creatures
        for i, creature_state in enumerate(self.play_areas[self.current_player]):
            allied_creatures = self.play_areas[self.current_player]
            enemy_creatures = self.play_areas[self.current_player.other()]
            keywords = creature_state.get_effective_keywords(
                allied_creatures, enemy_creatures
            )
            
            creature_id = id(creature_state)
            can_attack = False
            
            # Check attack eligibility
            if creature_id not in self.creatures_attacked_this_turn:
                can_attack = True
            elif Keyword.FRENZY in keywords and creature_state.attack_count == 1:
                # FRENZY second attack
                can_attack = True
            
            if can_attack:
                actions.append(
                    Action(ActionType.ATTACK, self.current_player, creature_index=i)
                )
        
        return actions

    def draw_cards(self, player: Player, count: int) -> int:
        # Draw cards from deck to hand
        cards_drawn = 0
        for _ in range(count):
            if not self.decks[player]:
                break  # Deck empty
            if len(self.hands[player]) < MAX_HAND_SIZE:
                self.hands[player].append(self.decks[player].pop())
                cards_drawn += 1
        return cards_drawn

    def discard_random(self, player: Player, count: int):
        # Discard random cards from hand
        count = min(count, len(self.hands[player]))
        if count > 0:
            discarded = random.sample(self.hands[player], count)
            for card in discarded:
                self.hands[player].remove(card)
                self.discard_piles[player].append(card)

    def update_creature_indices(self):
        # Update creature position indices
        for player in [Player.PLAYER_1, Player.PLAYER_2]:
            for i, creature in enumerate(self.play_areas[player]):
                creature.play_area_index = i

    def validate_state(self) -> List[str]:
        # Validate state consistency for debugging
        errors = []
        
        # Life totals
        for player, life in self.life.items():
            if life < 0:
                errors.append(f"{player.name} has negative life: {life}")
        
        # Mindbug counts
        for player, mindbugs in self.mindbugs_available.items():
            if mindbugs < 0 or mindbugs > MINDBUGS_PER_PLAYER:
                errors.append(f"{player.name} has invalid Mindbug count: {mindbugs}")
        
        # Creature indices
        for player, creatures in self.play_areas.items():
            for i, creature in enumerate(creatures):
                if creature.play_area_index != i:
                    errors.append(f"Creature index mismatch for {creature.card.name}")
                if creature.attack_count < 0:
                    errors.append(f"Negative attack count for {creature.card.name}")
        
        # Pending states
        if self.mindbug_decision_pending and not self.pending_creature_play:
            errors.append("Mindbug decision pending but no creature play pending")
        
        if self.hunter_choice_pending and not self.valid_blockers:
            errors.append("Hunter choice pending but no valid blockers")
        
        return errors