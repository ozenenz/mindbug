#!/usr/bin/env python3
"""Train Deep CFR on Mindbug."""
import argparse
import sys
from pathlib import Path

from mindbug.training import Trainer
from mindbug.utils import (
    get_debug_config,
    get_distributed_config,
    get_performance_config,
    get_quick_config,
)


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train Deep CFR on Mindbug First Contact",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run
  python train.py --config quick --iterations 100
  
  # Production training
  python train.py --config performance --iterations 10000
  
  # Distributed training
  python train.py --config distributed --iterations 50000
  
  # Resume from checkpoint
  python train.py --resume checkpoints/run_20240101/checkpoint_iter_5000.pt
        """,
    )

    parser.add_argument(
        "--config",
        choices=["quick", "performance", "distributed", "debug"],
        default="quick",
        help="Configuration preset to use",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of training iterations (default depends on config)",
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints", help="Directory to save checkpoints"
    )
    parser.add_argument("--run-name", default=None, help="Name for this training run")
    parser.add_argument(
        "--checkpoint-interval", type=int, default=None, help="Save checkpoint every N iterations"
    )
    parser.add_argument(
        "--eval-interval", type=int, default=None, help="Run evaluation every N iterations"
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint")

    args = parser.parse_args()

    # Load configuration
    configs = {
        "quick": get_quick_config(),
        "performance": get_performance_config(),
        "distributed": get_distributed_config(),
        "debug": get_debug_config(),
    }

    config = configs[args.config]

    # Set defaults based on config
    defaults = {
        "quick": {
            "iterations": 100,
            "checkpoint_interval": 50,
            "eval_interval": 25,
        },
        "performance": {
            "iterations": 10000,
            "checkpoint_interval": 1000,
            "eval_interval": 500,
        },
        "distributed": {
            "iterations": 50000,
            "checkpoint_interval": 5000,
            "eval_interval": 1000,
        },
        "debug": {
            "iterations": 10,
            "checkpoint_interval": 5,
            "eval_interval": 5,
        },
    }

    preset = defaults[args.config]
    iterations = args.iterations or preset["iterations"]
    checkpoint_interval = args.checkpoint_interval or preset["checkpoint_interval"]
    eval_interval = args.eval_interval or preset["eval_interval"]

    # Check GPU availability
    import torch

    if config["use_gpu"] and not torch.cuda.is_available():
        print("WARNING: GPU requested but not available. Training will be slow!")
        response = input("Continue with CPU? (y/n): ")
        if response.lower() != "y":
            sys.exit(1)
        config["use_gpu"] = False

    # Create trainer
    trainer = Trainer(config=config, run_name=args.run_name, checkpoint_dir=args.checkpoint_dir)

    # Resume if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.cfr.load_checkpoint(args.resume)

    # Train
    print(f"\nStarting {args.config} training for {iterations} iterations")
    trainer.train(
        num_iterations=iterations,
        checkpoint_interval=checkpoint_interval,
        eval_interval=eval_interval,
    )


if __name__ == "__main__":
    main()
