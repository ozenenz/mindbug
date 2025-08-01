"""Configuration presets for Deep CFR training."""

from typing import Any, Dict


def get_base_config() -> Dict[str, Any]:
    """Base configuration with common settings."""
    return {
        # Hardware
        "use_gpu": True,
        # Network architecture
        "num_card_types": 32,
        "card_embedding_dim": 128,
        "hidden_dim": 256,
        # Optimization
        "learning_rate": 1e-3,
        "strategy_learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        # CFR settings
        "use_linear_cfr": True,
        # Logging
        "log_interval": 100,
    }


def get_quick_config() -> Dict[str, Any]:
    """Quick test configuration for rapid iteration."""
    config = get_base_config()
    config.update(
        {
            # Smaller network
            "card_embedding_dim": 64,
            "hidden_dim": 128,
            # Small batches and buffers
            "batch_size": 256,
            "strategy_batch_size": 1000,
            "buffer_size": 50000,
            # Minimal training
            "traversals_per_iteration": 100,
            "value_epochs": 50,
            "strategy_epochs": 100,
            "strategy_interval": 10,
        }
    )
    return config


def get_performance_config() -> Dict[str, Any]:
    """Production configuration for optimal performance."""
    config = get_base_config()
    config.update(
        {
            # Optimized batch sizes
            "batch_size": 2048,
            "strategy_batch_size": 10000,
            # Large buffers (2M as recommended)
            "buffer_size": 2000000,
            # Full training schedule
            "traversals_per_iteration": 1000,
            "value_epochs": 100,
            "strategy_epochs": 200,
            "strategy_interval": 10,
        }
    )
    return config


def get_distributed_config() -> Dict[str, Any]:
    """Configuration for multi-GPU training."""
    config = get_performance_config()
    config.update(
        {
            # Larger batches for multi-GPU
            "batch_size": 10240,
            "strategy_batch_size": 50000,
            # More traversals
            "traversals_per_iteration": 5000,
            # Huge buffers
            "buffer_size": 10000000,
        }
    )
    return config


def get_debug_config() -> Dict[str, Any]:
    """Minimal configuration for debugging."""
    config = get_base_config()
    config.update(
        {
            # CPU only
            "use_gpu": False,
            # Tiny settings
            "card_embedding_dim": 32,
            "hidden_dim": 64,
            "batch_size": 32,
            "strategy_batch_size": 100,
            "buffer_size": 1000,
            # Minimal training
            "traversals_per_iteration": 10,
            "value_epochs": 10,
            "strategy_epochs": 10,
            "strategy_interval": 5,
            # Frequent logging
            "log_interval": 1,
        }
    )
    return config
