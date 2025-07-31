import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from mindbug.algorithms.deep_cfr import DeepCFR
from mindbug.training.evaluate import Evaluator
from mindbug.utils.config import (
    get_quick_test_config,
    get_performance_config,
    get_distributed_config,
    get_debug_config
)
from mindbug.game import MindbugEngine, Player
from mindbug.test_framework import TestCardImplementations, TestCombatResolution


class TrainingMonitor:
    # Tracks and logs training progress with TensorBoard
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.start_time = time.time()
        self.iteration_times = []
        self.gpu_memory_usage = []
        self.buffer_sizes = []
        self.exploitability_history = []
    
    def log_iteration(self, iteration: int, cfr: DeepCFR, elapsed_time: float):
        # Log per-iteration metrics
        self.iteration_times.append(elapsed_time)
        avg_time = np.mean(self.iteration_times[-100:])
        self.writer.add_scalar('Performance/IterationTime', elapsed_time, iteration)
        self.writer.add_scalar('Performance/AvgIterationTime', avg_time, iteration)
        
        # Buffer sizes
        p1_adv = len(cfr.advantage_buffers[Player.PLAYER_1])
        p2_adv = len(cfr.advantage_buffers[Player.PLAYER_2])
        p1_str = len(cfr.strategy_buffers[Player.PLAYER_1])
        p2_str = len(cfr.strategy_buffers[Player.PLAYER_2])
        
        self.writer.add_scalar('Buffers/P1_Advantage', p1_adv, iteration)
        self.writer.add_scalar('Buffers/P2_Advantage', p2_adv, iteration)
        self.writer.add_scalar('Buffers/P1_Strategy', p1_str, iteration)
        self.writer.add_scalar('Buffers/P2_Strategy', p2_str, iteration)
        
        # GPU memory tracking
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            self.writer.add_scalar('Performance/GPU_Memory_MB', memory_mb, iteration)
            self.gpu_memory_usage.append(memory_mb)
    
    def log_evaluation(self, iteration: int, exploitability: float, 
                      self_play_results: Dict[str, float]):
        # Log evaluation metrics
        self.writer.add_scalar('Evaluation/Exploitability', exploitability, iteration)
        self.exploitability_history.append((iteration, exploitability))
        
        self.writer.add_scalar('Evaluation/P1_WinRate', self_play_results['agent1_win_rate'], iteration)
        self.writer.add_scalar('Evaluation/P2_WinRate', self_play_results['agent2_win_rate'], iteration)
        self.writer.add_scalar('Evaluation/DrawRate', self_play_results['draw_rate'], iteration)
    
    def log_network_stats(self, iteration: int, cfr: DeepCFR):
        # Log network gradient norms and weights
        for player in [Player.PLAYER_1, Player.PLAYER_2]:
            if player in cfr.value_networks:
                network = cfr.value_networks[player]
                
                # Calculate gradient norm
                total_norm = 0.0
                for p in network.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                player_name = "P1" if player == Player.PLAYER_1 else "P2"
                self.writer.add_scalar(f'Network/{player_name}_GradNorm', total_norm, iteration)
                
                # Weight histograms
                for name, param in network.named_parameters():
                    self.writer.add_histogram(f'Weights/{player_name}/{name}', param, iteration)
    
    def print_summary(self, iteration: int):
        # Print training summary to console
        elapsed = time.time() - self.start_time
        iterations_per_sec = iteration / elapsed
        
        print(f"\n{'='*60}")
        print(f"Training Summary - Iteration {iteration}")
        print(f"{'='*60}")
        print(f"Total time: {elapsed/3600:.2f} hours")
        print(f"Speed: {iterations_per_sec:.2f} iterations/second")
        
        if self.gpu_memory_usage:
            print(f"GPU memory: {np.mean(self.gpu_memory_usage[-100:]):.1f} MB (avg)")
        
        if self.exploitability_history:
            recent_exploit = self.exploitability_history[-1][1]
            print(f"Exploitability: {recent_exploit:.4f}")
            
            # Check convergence
            if len(self.exploitability_history) >= 5:
                recent_values = [e[1] for e in self.exploitability_history[-5:]]
                if max(recent_values) - min(recent_values) < 0.001:
                    print("✓ Algorithm appears to be converging!")
        
        print(f"{'='*60}\n")
    
    def close(self):
        self.writer.close()


class DistributedTrainer:
    # Handles distributed training across multiple GPUs
    def __init__(self, config: dict, world_size: int, rank: int):
        self.config = config
        self.world_size = world_size
        self.rank = rank
        
        # Initialize distributed backend
        if world_size > 1:
            torch.distributed.init_process_group(
                backend='nccl',
                world_size=world_size,
                rank=rank
            )
    
    def synchronize_gradients(self, network: torch.nn.Module):
        # Average gradients across all processes
        if self.world_size > 1:
            for param in network.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(param.grad.data)
                    param.grad.data /= self.world_size
    
    def gather_samples(self, local_samples: List) -> List:
        # Gather samples from all processes
        if self.world_size == 1:
            return local_samples
        
        # All-gather samples
        all_samples = [None] * self.world_size
        torch.distributed.all_gather_object(all_samples, local_samples)
        
        # Flatten list
        return [sample for samples in all_samples for sample in samples]


def validate_implementation():
    # Run validation tests before training
    print("Running implementation validation tests...")
    
    # Test critical cards
    test_cards = TestCardImplementations()
    test_cards.setUp()
    
    try:
        test_cards.test_sharky_crab_dog_mummypus()
        print("✓ Sharky Crab-Dog-Mummypus implementation correct")
    except AssertionError as e:
        print(f"✗ Sharky Crab-Dog-Mummypus FAILED: {e}")
        return False
    
    try:
        test_cards.test_deathweaver_blocks_play_effects()
        print("✓ Deathweaver implementation correct")
    except AssertionError as e:
        print(f"✗ Deathweaver FAILED: {e}")
        return False
    
    try:
        test_cards.test_elephantopus_blocking_restriction()
        print("✓ Elephantopus implementation correct")
    except AssertionError as e:
        print(f"✗ Elephantopus FAILED: {e}")
        return False
    
    # Test combat
    test_combat = TestCombatResolution()
    test_combat.setUp()
    
    try:
        test_combat.test_tough_vs_poisonous()
        print("✓ TOUGH vs POISONOUS interaction correct")
    except AssertionError as e:
        print(f"✗ TOUGH vs POISONOUS FAILED: {e}")
        return False
    
    print("\nAll validation tests passed! ✓")
    return True


def train_with_monitoring(
    config_name: str = "quick",
    num_iterations: Optional[int] = None,
    checkpoint_dir: str = "checkpoints",
    validate: bool = True,
    distributed: bool = False,
    resume_from: Optional[str] = None
):
    # Main training function with monitoring
    
    # Load configuration
    configs = {
        "quick": get_quick_test_config(),
        "performance": get_performance_config(),
        "distributed": get_distributed_config(),
        "debug": get_debug_config()
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}")
    
    config = configs[config_name]
    
    # Set default iterations
    if num_iterations is None:
        defaults = {"quick": 100, "performance": 10000, "distributed": 50000, "debug": 10}
        num_iterations = defaults.get(config_name, 1000)
    
    # Validate implementation
    if validate and not validate_implementation():
        print("\nImplementation validation failed! Fix bugs before training.")
        return None
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config_name}_{timestamp}"
    run_dir = Path(checkpoint_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Initialize monitoring
    monitor = TrainingMonitor(str(run_dir / "tensorboard"))
    
    # Initialize Deep CFR
    print(f"\nInitializing Deep CFR with {config_name} configuration...")
    print(f"Target iterations: {num_iterations}")
    print(f"Output directory: {run_dir}")
    
    cfr = DeepCFR(config)
    
    # Resume from checkpoint
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        cfr.load_checkpoint(resume_from)
    
    # Training loop
    print("\nStarting training...")
    checkpoint_interval = max(1, num_iterations // 10)
    eval_interval = max(1, num_iterations // 20)
    
    try:
        for i in range(cfr.iteration_count + 1, num_iterations + 1):
            # Time iteration
            iter_start = time.time()
            
            # Train one iteration
            cfr.train(num_iterations=1)
            
            iter_time = time.time() - iter_start
            
            # Log metrics
            monitor.log_iteration(i, cfr, iter_time)
            
            # Periodic evaluation
            if i % eval_interval == 0:
                print(f"\nEvaluating at iteration {i}...")
                
                # Self-play evaluation
                results = Evaluator.evaluate_agents(cfr, cfr, num_games=100)
                
                # Exploitability estimation
                exploitability = Evaluator.compute_exploitability(cfr, num_iterations=100)
                
                monitor.log_evaluation(i, exploitability, results)
                monitor.log_network_stats(i, cfr)
                
                print(f"Exploitability: {exploitability:.4f}")
                print(f"Self-play win rates - P1: {results['agent1_win_rate']:.2%}, "
                      f"P2: {results['agent2_win_rate']:.2%}, "
                      f"Draw: {results['draw_rate']:.2%}")
            
            # Checkpointing
            if i % checkpoint_interval == 0:
                checkpoint_path = run_dir / f"checkpoint_iter_{i}.pt"
                cfr.save_checkpoint(str(checkpoint_path))
                print(f"Saved checkpoint: {checkpoint_path}")
                
                # Print summary
                monitor.print_summary(i)
            
            # Progress bar
            if i % max(1, num_iterations // 100) == 0:
                progress = i / num_iterations * 100
                eta = (num_iterations - i) * np.mean(monitor.iteration_times[-100:] or [iter_time])
                print(f"Progress: {progress:.1f}% | ETA: {eta/3600:.1f} hours", end='\r')
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    finally:
        # Save final checkpoint
        final_path = run_dir / "final_checkpoint.pt"
        cfr.save_checkpoint(str(final_path))
        print(f"\nSaved final checkpoint: {final_path}")
        
        # Save training history
        history = {
            "config": config,
            "iterations": cfr.iteration_count,
            "exploitability_history": monitor.exploitability_history,
            "training_time": time.time() - monitor.start_time
        }
        
        with open(run_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        monitor.close()
        
        print(f"\nTraining complete! Results saved to: {run_dir}")
        
        # Final summary
        monitor.print_summary(cfr.iteration_count)
    
    return cfr, run_dir


def main():
    # Command-line interface
    parser = argparse.ArgumentParser(
        description="Train Deep CFR on Mindbug First Contact",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run
  python train_improved.py --config quick --iterations 100
  
  # Production training
  python train_improved.py --config performance --iterations 10000
  
  # Distributed training on multiple GPUs
  python train_improved.py --config distributed --distributed
  
  # Resume from checkpoint
  python train_improved.py --config performance --resume checkpoints/run_20240730_120000/checkpoint_iter_5000.pt
        """
    )
    
    parser.add_argument(
        "--config",
        choices=["quick", "performance", "distributed", "debug"],
        default="quick",
        help="Configuration preset to use"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of training iterations (default depends on config)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip implementation validation tests"
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint"
    )
    
    args = parser.parse_args()
    
    # Check GPU availability
    if args.config in ["performance", "distributed"] and not torch.cuda.is_available():
        print("WARNING: GPU not available. Training will be slow!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Run training
    train_with_monitoring(
        config_name=args.config,
        num_iterations=args.iterations,
        checkpoint_dir=args.checkpoint_dir,
        validate=not args.no_validate,
        distributed=args.distributed,
        resume_from=args.resume
    )


if __name__ == "__main__":
    main()