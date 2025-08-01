"""Training orchestration with monitoring."""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ..algorithms import DeepCFR
from ..core import Player
from .evaluator import Evaluator


class TrainingMonitor:
    """Tracks and logs training progress."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.start_time = time.time()
        self.iteration_times: List[float] = []
        self.exploitability_history: List[Tuple[int, float]] = []

    def log_iteration(self, iteration: int, cfr: DeepCFR, elapsed_time: float) -> None:
        """Log per-iteration metrics."""
        self.iteration_times.append(elapsed_time)

        # Performance metrics
        avg_time = np.mean(self.iteration_times[-100:]) if self.iteration_times else elapsed_time
        self.writer.add_scalar("Performance/IterationTime", elapsed_time, iteration)
        self.writer.add_scalar("Performance/AvgIterationTime", avg_time, iteration)

        # Buffer sizes
        for player in [Player.PLAYER_1, Player.PLAYER_2]:
            adv_size = len(cfr.advantage_buffers[player])
            str_size = len(cfr.strategy_buffers[player])

            player_name = "P1" if player == Player.PLAYER_1 else "P2"
            self.writer.add_scalar(f"Buffers/{player_name}_Advantage", adv_size, iteration)
            self.writer.add_scalar(f"Buffers/{player_name}_Strategy", str_size, iteration)

        # GPU memory
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            self.writer.add_scalar("Performance/GPU_Memory_MB", memory_mb, iteration)

    def log_evaluation(
        self, iteration: int, exploitability: float, self_play_results: Dict[str, float]
    ) -> None:
        """Log evaluation metrics."""
        self.writer.add_scalar("Evaluation/Exploitability", exploitability, iteration)
        self.exploitability_history.append((iteration, exploitability))

        self.writer.add_scalar(
            "Evaluation/P1_WinRate", self_play_results["agent1_win_rate"], iteration
        )
        self.writer.add_scalar(
            "Evaluation/P2_WinRate", self_play_results["agent2_win_rate"], iteration
        )
        self.writer.add_scalar("Evaluation/DrawRate", self_play_results["draw_rate"], iteration)

    def print_summary(self, iteration: int) -> None:
        """Print training summary to console."""
        elapsed = time.time() - self.start_time
        iterations_per_sec = iteration / elapsed if elapsed > 0 else 0

        print(f"\n{'='*60}")
        print(f"Training Summary - Iteration {iteration}")
        print(f"{'='*60}")
        print(f"Total time: {elapsed/3600:.2f} hours")
        print(f"Speed: {iterations_per_sec:.2f} iterations/second")

        if self.exploitability_history:
            recent_exploit = self.exploitability_history[-1][1]
            print(f"Exploitability: {recent_exploit:.4f}")

            # Check convergence
            if len(self.exploitability_history) >= 5:
                recent_values = [e[1] for e in self.exploitability_history[-5:]]
                if max(recent_values) - min(recent_values) < 0.001:
                    print("✓ Algorithm appears to be converging!")

        print(f"{'='*60}\n")

    def close(self) -> None:
        """Close the tensorboard writer."""
        self.writer.close()


class Trainer:
    """Manages Deep CFR training."""

    def __init__(
        self, config: dict, run_name: Optional[str] = None, checkpoint_dir: str = "checkpoints"
    ):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)

        # Create run directory
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"

        self.run_dir = self.checkpoint_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Initialize components
        self.cfr = DeepCFR(config)
        self.monitor = TrainingMonitor(str(self.run_dir / "tensorboard"))
        self.evaluator = Evaluator()

    def train(
        self,
        num_iterations: int,
        checkpoint_interval: int = 1000,
        eval_interval: int = 500,
        save_final: bool = True,
    ) -> DeepCFR:
        """Run training loop."""
        print(f"Starting training for {num_iterations} iterations")
        print(f"Output directory: {self.run_dir}")

        try:
            for i in range(1, num_iterations + 1):
                iter_start = time.time()

                # Train one iteration
                self.cfr.train(num_iterations=1)

                iter_time = time.time() - iter_start
                self.monitor.log_iteration(i, self.cfr, iter_time)

                # Evaluation
                if i % eval_interval == 0:
                    self._evaluate(i)

                # Checkpointing
                if i % checkpoint_interval == 0:
                    self._save_checkpoint(i)
                    self.monitor.print_summary(i)

                # Progress
                if i % max(1, num_iterations // 100) == 0:
                    progress = i / num_iterations * 100
                    eta = (num_iterations - i) * np.mean(self.monitor.iteration_times[-100:])
                    print(f"Progress: {progress:.1f}% | ETA: {eta/3600:.1f} hours", end="\r")

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")

        finally:
            if save_final:
                self._save_checkpoint(self.cfr.iteration_count, final=True)

            self._save_training_history()
            self.monitor.close()

            print(f"\nTraining complete! Results saved to: {self.run_dir}")
            self.monitor.print_summary(self.cfr.iteration_count)

        return self.cfr

    def _evaluate(self, iteration: int) -> None:
        """Run evaluation."""
        print(f"\nEvaluating at iteration {iteration}...")

        # Self-play
        results = self.evaluator.self_play_evaluation(self.cfr, num_games=100)

        # Exploitability
        exploitability = self.evaluator.compute_exploitability(self.cfr, num_iterations=100)

        self.monitor.log_evaluation(iteration, exploitability, results)

        print(f"Exploitability: {exploitability:.4f}")
        print(
            f"Self-play - P1: {results['agent1_win_rate']:.2%}, "
            f"P2: {results['agent2_win_rate']:.2%}, "
            f"Draw: {results['draw_rate']:.2%}"
        )

    def _save_checkpoint(self, iteration: int, final: bool = False) -> None:
        """Save model checkpoint."""
        if final:
            path = self.run_dir / "final_checkpoint.pt"
        else:
            path = self.run_dir / f"checkpoint_iter_{iteration}.pt"

        self.cfr.save_checkpoint(str(path))
        print(f"\nSaved checkpoint: {path}")

    def _save_training_history(self) -> None:
        """Save training history."""
        history = {
            "config": self.config,
            "iterations": self.cfr.iteration_count,
            "exploitability_history": self.monitor.exploitability_history,
            "training_time": time.time() - self.monitor.start_time,
        }

        with open(self.run_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
