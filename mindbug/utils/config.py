from typing import Any, Dict


def get_quick_test_config() -> Dict[str, Any]:
    # Quick test configuration for rapid iteration
    return {
        # GPU settings
        "use_gpu": True,
        
        # Network architecture
        "card_embedding_dim": 64,  # Smaller for quick tests
        "hidden_dim": 128,
        "num_card_types": 32,
        
        # Training hyperparameters
        "learning_rate": 1e-3,
        "strategy_learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        
        # Batch sizes
        "batch_size": 256,
        "strategy_batch_size": 1000,
        
        # Buffer sizes (smaller for quick tests)
        "buffer_size": 50000,
        
        # CFR settings
        "traversals_per_iteration": 100,  # K traversals per iteration
        "use_linear_cfr": True,
        
        # Training schedule
        "value_epochs": 50,
        "strategy_epochs": 100,
        "strategy_interval": 10,  # Train strategy every N iterations
        
        # Logging
        "log_interval": 10,
        "checkpoint_interval": 50,
    }


def get_performance_config() -> Dict[str, Any]:
    """Production configuration for optimal performance."""
    return {
        # GPU settings
        "use_gpu": True,
        
        # Network architecture (7-layer dual-branch)
        "card_embedding_dim": 128,
        "hidden_dim": 256,  # Results in ~98k parameters total
        "num_card_types": 32,
        
        # Training hyperparameters (from paper)
        "learning_rate": 1e-3,
        "strategy_learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        
        # Batch sizes (optimized for GPU memory)
        "batch_size": 2048,  # Can go up to 20480 with enough GPU memory
        "strategy_batch_size": 10000,
        
        # Buffer sizes (2M as recommended)
        "buffer_size": 2000000,
        
        # CFR settings
        "traversals_per_iteration": 1000,  # K=1000 for complex games
        "use_linear_cfr": True,  # Linear CFR for better convergence
        
        # Training schedule
        "value_epochs": 100,  # 750-4000 SGD updates
        "strategy_epochs": 200,  # 5000-20000 SGD updates
        "strategy_interval": 10,
        
        # Logging
        "log_interval": 100,
        "checkpoint_interval": 1000,
    }


def get_distributed_config() -> Dict[str, Any]:
    """Configuration for distributed training across multiple GPUs."""
    config = get_performance_config()
    config.update({
        # Larger batch sizes for multi-GPU
        "batch_size": 10240,
        "strategy_batch_size": 50000,
        
        # More traversals per iteration
        "traversals_per_iteration": 5000,
        
        # Larger buffers
        "buffer_size": 10000000,  # 10M samples
        
        # Mixed precision training
        "use_mixed_precision": True,
        
        # Distributed settings
        "num_workers": 8,
        "sync_gradients_interval": 10,
    })
    return config


def get_debug_config() -> Dict[str, Any]:
    """Minimal configuration for debugging."""
    return {
        # CPU only for debugging
        "use_gpu": False,
        
        # Tiny network
        "card_embedding_dim": 32,
        "hidden_dim": 64,
        "num_card_types": 32,
        
        # Small batches
        "batch_size": 32,
        "strategy_batch_size": 100,
        
        # Tiny buffers
        "buffer_size": 1000,
        
        # Minimal training
        "traversals_per_iteration": 10,
        "use_linear_cfr": False,
        "value_epochs": 10,
        "strategy_epochs": 10,
        "strategy_interval": 5,
        
        # Frequent logging
        "log_interval": 1,
        "checkpoint_interval": 10,
        
        # Debug settings
        "learning_rate": 1e-2,
        "strategy_learning_rate": 1e-3,
        "weight_decay": 0.0,
        "grad_clip": 10.0,
    }