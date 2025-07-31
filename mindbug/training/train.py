import argparse
import json
import os
from datetime import datetime

from ..algorithms.deep_cfr import DeepCFR
from ..utils.config import get_performance_config, get_quick_test_config
from .evaluate import Evaluator


def train(
    config_name: str = "quick",
    num_iterations: int = None,
    checkpoint_dir: str = "checkpoints",
):
    # Basic training function
    
    # Load configuration
    if config_name == "quick":
        config = get_quick_test_config()
        default_iterations = 100
    else:
        config = get_performance_config()
        default_iterations = 10000
    
    if num_iterations is None:
        num_iterations = default_iterations
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(checkpoint_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Initializing Deep CFR with {config_name} configuration...")
    deep_cfr = DeepCFR(config)
    
    # Set intervals
    checkpoint_interval = max(1, num_iterations // 10)
    eval_interval = max(1, num_iterations // 20)
    
    print(f"Starting training for {num_iterations} iterations...")
    
    # Training loop
    for i in range(0, num_iterations, checkpoint_interval):
        iterations_to_train = min(checkpoint_interval, num_iterations - i)
        deep_cfr.train(iterations_to_train)
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            run_dir, f"checkpoint_{i + iterations_to_train}.pt"
        )
        deep_cfr.save_checkpoint(checkpoint_path)
        print(f"Saved checkpoint at iteration {i + iterations_to_train}")
        
        # Evaluate periodically
        if (i + iterations_to_train) % eval_interval == 0:
            print("Evaluating...")
            
            # Self-play evaluation
            results = Evaluator.evaluate_agents(deep_cfr, deep_cfr, num_games=100)
            print(f"Self-play results: {results}")
            
            # Exploitability estimate
            exploitability = Evaluator.compute_exploitability(
                deep_cfr, num_iterations=100
            )
            print(f"Estimated exploitability: {exploitability:.4f}")
            
            # Save evaluation results
            eval_results = {
                "iteration": i + iterations_to_train,
                "self_play": results,
                "exploitability": exploitability,
            }
            with open(
                os.path.join(run_dir, f"eval_{i + iterations_to_train}.json"), "w"
            ) as f:
                json.dump(eval_results, f, indent=2)
    
    print(f"Training complete! Results saved to {run_dir}")
    return deep_cfr, run_dir


def main():
    # Command-line interface
    parser = argparse.ArgumentParser(description="Train Deep CFR on Mindbug")
    parser.add_argument(
        "--config",
        choices=["quick", "performance"],
        default="quick",
        help="Configuration to use",
    )
    parser.add_argument(
        "--iterations", type=int, default=None, help="Number of training iterations"
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints", help="Directory to save checkpoints"
    )
    args = parser.parse_args()
    
    train(
        config_name=args.config,
        num_iterations=args.iterations,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()