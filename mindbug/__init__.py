# Mindbug First Contact - Deep CFR Implementation
__version__ = "1.0.0"

# Core game components
from .game import MindbugEngine, GameState, Player

# Deep CFR algorithm
from .algorithms import DeepCFR

__all__ = ["MindbugEngine", "GameState", "Player", "DeepCFR"]