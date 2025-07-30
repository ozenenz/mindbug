from dataclasses import dataclass
from typing import Optional

from .cards import Card
from .constants import ActionType, Player


@dataclass
class Action:
    action_type: ActionType
    player: Player
    card: Optional[Card] = None
    creature_index: Optional[int] = None
    target_index: Optional[int] = None

    def __hash__(self):
        return hash(
            (
                self.action_type,
                self.player,
                self.card,
                self.creature_index,
                self.target_index,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return (
            self.action_type == other.action_type
            and self.player == other.player
            and self.card == other.card
            and self.creature_index == other.creature_index
            and self.target_index == other.target_index
        )

    def __repr__(self):
        parts = [f"Action({self.action_type.name}", f"player={self.player.name}"]
        if self.card:
            parts.append(f"card={self.card.name}")
        if self.creature_index is not None:
            parts.append(f"creature_idx={self.creature_index}")
        if self.target_index is not None:
            parts.append(f"target_idx={self.target_index}")
        return ", ".join(parts) + ")"
