"""Game state representation."""

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .actions import Action
from .cards import Card
from .constants import (
    MINDBUGS_PER_PLAYER,
    STARTING_HAND_SIZE,
    STARTING_LIFE,
    ActionType,
    Keyword,
    Player,
)
from .creature import CreatureState


@dataclass
class GameState:
    """Complete game state representation."""

    # Core state
    current_player: Player
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
        default_factory=lambda: {
            Player.PLAYER_1: [],
            Player.PLAYER_2: [],
        }
    )
    decks: Dict[Player, List[Card]] = field(
        default_factory=lambda: {
            Player.PLAYER_1: [],
            Player.PLAYER_2: [],
        }
    )
    play_areas: Dict[Player, List[CreatureState]] = field(
        default_factory=lambda: {
            Player.PLAYER_1: [],
            Player.PLAYER_2: [],
        }
    )
    discard_piles: Dict[Player, List[Card]] = field(
        default_factory=lambda: {
            Player.PLAYER_1: [],
            Player.PLAYER_2: [],
        }
    )

    # Turn state
    extra_turn_pending: Optional[Player] = None
    creatures_attacked_this_turn: Set[int] = field(default_factory=set)

    # Special phases
    mindbug_decision_pending: bool = False
    pending_creature_play: Optional[Tuple[Player, Card]] = None

    hunter_choice_pending: bool = False
    valid_blockers: List[int] = field(default_factory=list)
    attacking_creature: Optional[CreatureState] = None

    # Passive effects tracking
    deathweaver_active: Dict[Player, bool] = field(
        default_factory=lambda: {
            Player.PLAYER_1: False,
            Player.PLAYER_2: False,
        }
    )

    def copy(self) -> "GameState":
        """Create deep copy of state."""
        return deepcopy(self)

    def is_terminal(self) -> bool:
        """Check if game has ended."""
        # Life loss
        if any(life <= 0 for life in self.life.values()):
            return True

        # No legal actions means loss (can't play or attack)
        if not self.mindbug_decision_pending and not self.hunter_choice_pending:
            legal_actions = self.get_legal_actions()
            if not legal_actions:
                return True

        return False

    def get_winner(self) -> Optional[Player]:
        """Determine winner if game is terminal."""
        if not self.is_terminal():
            return None

        # Check life totals
        p1_life = self.life[Player.PLAYER_1]
        p2_life = self.life[Player.PLAYER_2]

        if p1_life <= 0 and p2_life <= 0:
            return None  # Draw
        elif p1_life <= 0:
            return Player.PLAYER_2
        elif p2_life <= 0:
            return Player.PLAYER_1

        # No actions available - current player loses
        return self.current_player.other()

    def get_legal_actions(self) -> List[Action]:
        """Get all legal actions for current game state."""
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

        # Normal turn - play creatures from hand
        for card in self.hands[self.current_player]:
            actions.append(Action(ActionType.PLAY_CREATURE, self.current_player, card=card))

        # Normal turn - attack with creatures
        for i, creature in enumerate(self.play_areas[self.current_player]):
            creature_id = id(creature)

            # Check if can attack
            can_attack = False
            if creature_id not in self.creatures_attacked_this_turn:
                can_attack = True
            else:
                # Check FRENZY for second attack
                allied = self.play_areas[self.current_player]
                enemy = self.play_areas[self.current_player.other()]
                keywords = creature.get_effective_keywords(allied, enemy)

                if Keyword.FRENZY in keywords and creature.attack_count == 1:
                    can_attack = True

            if can_attack:
                actions.append(Action(ActionType.ATTACK, self.current_player, creature_index=i))

        return actions

    def draw_cards(self, player: Player, count: int) -> int:
        """Draw cards from deck to hand. Returns number drawn."""
        cards_drawn = 0
        for _ in range(count):
            if not self.decks[player]:
                break
            self.hands[player].append(self.decks[player].pop())
            cards_drawn += 1
        return cards_drawn

    def discard_random(self, player: Player, count: int) -> List[Card]:
        """Discard random cards from hand. Returns discarded cards."""
        discarded = []
        count = min(count, len(self.hands[player]))

        if count > 0:
            cards = random.sample(self.hands[player], count)
            for card in cards:
                self.hands[player].remove(card)
                self.discard_piles[player].append(card)
                discarded.append(card)

        return discarded

    def update_deathweaver_status(self) -> None:
        """Update which players are affected by Deathweaver."""
        # Player 1 is affected if Player 2 has Deathweaver
        self.deathweaver_active[Player.PLAYER_1] = any(
            c.card.name == "Deathweaver" for c in self.play_areas[Player.PLAYER_2]
        )

        # Player 2 is affected if Player 1 has Deathweaver
        self.deathweaver_active[Player.PLAYER_2] = any(
            c.card.name == "Deathweaver" for c in self.play_areas[Player.PLAYER_1]
        )

    def reset_attack_counts(self) -> None:
        """Reset attack counts for all creatures."""
        for player in [Player.PLAYER_1, Player.PLAYER_2]:
            for creature in self.play_areas[player]:
                creature.attack_count = 0

    def __str__(self) -> str:
        """String representation for debugging."""
        lines = [
            f"Current Player: {self.current_player.name}",
            f"Life - P1: {self.life[Player.PLAYER_1]}, P2: {self.life[Player.PLAYER_2]}",
            f"Mindbugs - P1: {self.mindbugs_available[Player.PLAYER_1]}, P2: {self.mindbugs_available[Player.PLAYER_2]}",
            f"Cards in hand - P1: {len(self.hands[Player.PLAYER_1])}, P2: {len(self.hands[Player.PLAYER_2])}",
            f"Creatures - P1: {len(self.play_areas[Player.PLAYER_1])}, P2: {len(self.play_areas[Player.PLAYER_2])}",
        ]

        if self.mindbug_decision_pending:
            lines.append("Phase: Mindbug Decision")
        elif self.hunter_choice_pending:
            lines.append("Phase: Hunter Choice")

        return "\n".join(lines)
