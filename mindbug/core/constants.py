"""Game constants and enumerations."""

from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self


class Player(Enum):
    """Represents the two players."""

    PLAYER_1 = 0
    PLAYER_2 = 1

    def other(self) -> "Player":
        """Get the opposing player."""
        return Player.PLAYER_2 if self == Player.PLAYER_1 else Player.PLAYER_1


class ActionType(Enum):
    """All possible action types in the game."""

    PLAY_CREATURE = auto()
    ATTACK = auto()
    USE_MINDBUG = auto()
    PASS_MINDBUG = auto()
    CHOOSE_BLOCKER = auto()


class Keyword(Enum):
    """Creature keyword abilities."""

    POISONOUS = "POISONOUS"  # Always defeats in combat
    SNEAKY = "SNEAKY"  # Can only be blocked by SNEAKY
    HUNTER = "HUNTER"  # Choose which creature blocks
    TOUGH = "TOUGH"  # Survives first defeat
    FRENZY = "FRENZY"  # Attack twice per turn


class TriggerType(Enum):
    """When abilities trigger."""

    PLAY = "Play"  # When creature enters play
    ATTACK = "Attack"  # When creature attacks
    DEFEATED = "Defeated"  # When creature is defeated
    PASSIVE = "Passive"  # Always active


# Game configuration
STARTING_LIFE = 3
STARTING_HAND_SIZE = 5
DECK_SIZE_PER_PLAYER = 10
MINDBUGS_PER_PLAYER = 2
FIRST_CONTACT_DECK_SIZE = 48
