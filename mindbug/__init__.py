"""Mindbug Deep CFR - A state-of-the-art implementation."""

__version__ = "1.0.0"

from .algorithms import DeepCFR
from .core import GameState, MindbugEngine, Player

__all__ = ["DeepCFR", "GameState", "MindbugEngine", "Player"]
