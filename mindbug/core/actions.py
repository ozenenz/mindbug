"""Action representation for game moves."""

from dataclasses import dataclass
from typing import Optional

from .cards import Card
from .constants import ActionType, Player


@dataclass(frozen=True)
class Action:
    """Immutable action that can be taken in the game."""

    action_type: ActionType
    player: Player
    card: Optional[Card] = None  # For PLAY_CREATURE
    creature_index: Optional[int] = None  # For ATTACK and combat
    target_index: Optional[int] = None  # For CHOOSE_BLOCKER

    def __str__(self) -> str:
        """Human-readable action description."""
        parts = [f"{self.action_type.name} by {self.player.name}"]

        if self.card:
            parts.append(f"card={self.card.name}")
        if self.creature_index is not None:
            parts.append(f"creature={self.creature_index}")
        if self.target_index is not None:
            parts.append(f"target={self.target_index}")

        return f"Action({', '.join(parts)})"
