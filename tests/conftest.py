"""Pytest configuration and shared fixtures."""

import random

import numpy as np
import pytest
import torch

from mindbug.algorithms import DeepCFR
from mindbug.core import CardDatabase, GameState, MindbugEngine
from mindbug.utils import get_debug_config


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def initial_state():
    """Provide a fresh initial game state."""
    return MindbugEngine.create_initial_state(shuffle=False)


@pytest.fixture
def debug_cfr():
    """Provide a Debug CFR instance."""
    config = get_debug_config()
    return DeepCFR(config)


@pytest.fixture
def all_cards():
    """Provide all card definitions."""
    return CardDatabase.get_all_cards()


@pytest.fixture
def sample_deck():
    """Provide a sample deck."""
    return CardDatabase.get_first_contact_deck()


# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")


# Performance tracking
@pytest.fixture
def benchmark_timer():
    """Simple timer for performance benchmarks."""
    import time

    class Timer:
        def __init__(self):
            self.times = []

        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, *args):
            elapsed = time.time() - self.start
            self.times.append(elapsed)

        def average(self):
            return sum(self.times) / len(self.times) if self.times else 0

        def total(self):
            return sum(self.times)

    return Timer()
