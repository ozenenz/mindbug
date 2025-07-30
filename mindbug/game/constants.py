from enum import Enum, auto


class Player(Enum):
    PLAYER_1 = 0
    PLAYER_2 = 1

    def other(self) -> "Player":
        return Player.PLAYER_1 if self == Player.PLAYER_2 else Player.PLAYER_2


class Zone(Enum):
    DECK = auto()
    HAND = auto()
    PLAY = auto()
    DISCARD = auto()


class ActionType(Enum):
    PLAY_CREATURE = auto()
    ATTACK = auto()
    USE_MINDBUG = auto()
    PASS_MINDBUG = auto()
    CHOOSE_BLOCKER = auto()


class Keyword(Enum):
    POISONOUS = "POISONOUS"
    SNEAKY = "SNEAKY"
    HUNTER = "HUNTER"
    TOUGH = "TOUGH"
    FRENZY = "FRENZY"


class TriggerType(Enum):
    PLAY = "Play"
    ATTACK = "Attack"
    DEFEATED = "Defeated"
    PASSIVE = "Passive"


STARTING_LIFE = 3
STARTING_HAND_SIZE = 5
DECK_SIZE_PER_PLAYER = 10
MINDBUGS_PER_PLAYER = 2
MAX_HAND_SIZE = 5
