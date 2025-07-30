from typing import Any, Dict


def get_quick_test_config() -> Dict[str, Any]:
    return {
        "use_gpu": True,
        "hidden_dim": 128,
        "num_layers": 3,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "buffer_size": 50000,
        "traversals_per_iteration": 100,
        "train_interval": 1,
        "num_epochs": 50,
        "grad_clip": 1.0,
        "use_linear_cfr": True,
        "log_interval": 10,
    }


def get_performance_config() -> Dict[str, Any]:
    return {
        "use_gpu": True,
        "hidden_dim": 512,
        "num_layers": 4,
        "learning_rate": 1e-4,
        "batch_size": 256,
        "buffer_size": 2000000,
        "traversals_per_iteration": 1000,
        "train_interval": 1,
        "num_epochs": 100,
        "grad_clip": 1.0,
        "use_linear_cfr": True,
        "log_interval": 100,
    }
