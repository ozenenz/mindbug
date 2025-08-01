#!/usr/bin/env python3
"""Benchmark script for Mindbug Deep CFR implementation."""
import argparse
import statistics
import time
from typing import Dict, List

import numpy as np
import torch

from mindbug import DeepCFR, GameState, MindbugEngine, Player
from mindbug.core import Action, ActionType
from mindbug.training import Evaluator
from mindbug.utils import get_performance_config, get_quick_config


class Benchmark:
    """Performance benchmarking utilities."""

    @staticmethod
    def benchmark_state_operations(num_iterations: int = 1000) -> Dict[str, float]:
        """Benchmark core state operations."""
        results = {}

        # Initial state creation
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            state = MindbugEngine.create_initial_state()
            times.append(time.perf_counter() - start)
        results["initial_state_creation"] = statistics.mean(times) * 1000  # ms

        # State copying
        state = MindbugEngine.create_initial_state()
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = state.copy()
            times.append(time.perf_counter() - start)
        results["state_copy"] = statistics.mean(times) * 1000  # ms

        # Legal action generation
        times = []
        for _ in range(100):  # Fewer iterations as this varies
            state = MindbugEngine.create_initial_state()
            # Play some moves to get varied states
            for _ in range(5):
                actions = state.get_legal_actions()
                if actions and not state.is_terminal():
                    action = np.random.choice(actions)
                    state = MindbugEngine.apply_action(state, action)

            start = time.perf_counter()
            _ = state.get_legal_actions()
            times.append(time.perf_counter() - start)
        results["legal_actions"] = statistics.mean(times) * 1000  # ms

        # Action application
        times = []
        for _ in range(100):
            state = MindbugEngine.create_initial_state()
            actions = state.get_legal_actions()
            if actions:
                action = actions[0]
                start = time.perf_counter()
                _ = MindbugEngine.apply_action(state, action)
                times.append(time.perf_counter() - start)
        results["apply_action"] = statistics.mean(times) * 1000  # ms

        return results

    @staticmethod
    def benchmark_neural_network(
        batch_sizes: List[int] = [1, 16, 256, 1024]
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark neural network performance."""
        from mindbug.algorithms import DualBranchNetwork, StateEncoder

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = DualBranchNetwork().to(device)
        network.eval()

        results = {}

        for batch_size in batch_sizes:
            # Create dummy batch
            card_indices = torch.randint(0, 32, (batch_size, 30)).to(device)
            history_features = torch.randn(batch_size, 64).to(device)

            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = network(card_indices, history_features)

            # Time forward pass
            if device.type == "cuda":
                torch.cuda.synchronize()

            times = []
            for _ in range(100):
                start = time.perf_counter()
                with torch.no_grad():
                    _ = network(card_indices, history_features)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

            avg_time = statistics.mean(times) * 1000  # ms
            throughput = batch_size / (avg_time / 1000)  # samples/sec

            results[f"batch_{batch_size}"] = {"time_ms": avg_time, "throughput": throughput}

        return results

    @staticmethod
    def benchmark_cfr_iteration(config: dict, num_iterations: int = 10) -> Dict[str, float]:
        """Benchmark CFR iteration performance."""
        cfr = DeepCFR(config)

        times = []
        for i in range(num_iterations):
            start = time.perf_counter()
            cfr.train(num_iterations=1)
            times.append(time.perf_counter() - start)

        return {
            "mean_time": statistics.mean(times),
            "std_time": statistics.stdev(times) if len(times) > 1 else 0,
            "min_time": min(times),
            "max_time": max(times),
            "traversals_per_second": config["traversals_per_iteration"] / statistics.mean(times),
        }

    @staticmethod
    def benchmark_game_length(num_games: int = 100) -> Dict[str, float]:
        """Benchmark typical game lengths."""
        game_lengths = []

        for _ in range(num_games):
            state = MindbugEngine.create_initial_state()
            moves = 0

            while not state.is_terminal() and moves < 500:
                actions = state.get_legal_actions()
                if not actions:
                    break

                action = np.random.choice(actions)
                state = MindbugEngine.apply_action(state, action)
                moves += 1

            game_lengths.append(moves)

        return {
            "mean_length": statistics.mean(game_lengths),
            "std_length": statistics.stdev(game_lengths),
            "min_length": min(game_lengths),
            "max_length": max(game_lengths),
            "median_length": statistics.median(game_lengths),
        }

    @staticmethod
    def validate_implementation(verbose: bool = False) -> bool:
        """Run validation checks on implementation."""
        checks_passed = 0
        checks_total = 0

        # Check 1: Initial state validity
        checks_total += 1
        try:
            state = MindbugEngine.create_initial_state()
            assert state.life[Player.PLAYER_1] == 3
            assert state.life[Player.PLAYER_2] == 3
            assert len(state.hands[Player.PLAYER_1]) == 5
            assert len(state.hands[Player.PLAYER_2]) == 5
            checks_passed += 1
            if verbose:
                print("✓ Initial state validation passed")
        except AssertionError:
            if verbose:
                print("✗ Initial state validation failed")

        # Check 2: Action generation
        checks_total += 1
        try:
            state = MindbugEngine.create_initial_state()
            actions = state.get_legal_actions()
            assert len(actions) > 0
            assert all(isinstance(a, Action) for a in actions)
            checks_passed += 1
            if verbose:
                print("✓ Action generation validation passed")
        except AssertionError:
            if verbose:
                print("✗ Action generation validation failed")

        # Check 3: Game termination
        checks_total += 1
        try:
            state = GameState(current_player=Player.PLAYER_1)
            state.life[Player.PLAYER_1] = 0
            assert state.is_terminal()
            assert state.get_winner() == Player.PLAYER_2
            checks_passed += 1
            if verbose:
                print("✓ Game termination validation passed")
        except AssertionError:
            if verbose:
                print("✗ Game termination validation failed")

        # Check 4: Neural network
        checks_total += 1
        try:
            from mindbug.algorithms import DualBranchNetwork

            network = DualBranchNetwork()
            card_indices = torch.randint(0, 32, (8, 30))
            history = torch.randn(8, 64)
            output = network(card_indices, history)
            assert output.shape == (8, 1)
            checks_passed += 1
            if verbose:
                print("✓ Neural network validation passed")
        except Exception:
            if verbose:
                print("✗ Neural network validation failed")

        # Check 5: CFR algorithm
        checks_total += 1
        try:
            config = get_quick_config()
            config["traversals_per_iteration"] = 10
            cfr = DeepCFR(config)
            cfr.train(num_iterations=1)
            assert cfr.iteration_count == 1
            checks_passed += 1
            if verbose:
                print("✓ CFR algorithm validation passed")
        except Exception:
            if verbose:
                print("✗ CFR algorithm validation failed")

        if verbose:
            print(f"\nValidation complete: {checks_passed}/{checks_total} checks passed")

        return checks_passed == checks_total


def format_results(results: dict, title: str) -> str:
    """Format benchmark results for display."""
    lines = [f"\n{title}", "=" * len(title)]

    for key, value in results.items():
        if isinstance(value, dict):
            lines.append(f"\n{key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.3f}")
                else:
                    lines.append(f"  {k}: {v}")
        elif isinstance(value, float):
            lines.append(f"{key}: {value:.3f}")
        else:
            lines.append(f"{key}: {value}")

    return "\n".join(lines)


def main():
    """Run benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark Mindbug Deep CFR implementation")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--state", action="store_true", help="Benchmark state operations")
    parser.add_argument("--network", action="store_true", help="Benchmark neural network")
    parser.add_argument("--cfr", action="store_true", help="Benchmark CFR iterations")
    parser.add_argument("--games", action="store_true", help="Benchmark game statistics")
    parser.add_argument("--validate", action="store_true", help="Validate implementation")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Default to validation if nothing specified
    if not any([args.all, args.state, args.network, args.cfr, args.games, args.validate]):
        args.validate = True

    print("Mindbug Deep CFR Benchmark Suite")
    print("================================")

    # Validation
    if args.validate or args.all:
        print("\nRunning validation checks...")
        valid = Benchmark.validate_implementation(verbose=args.verbose)
        if not valid:
            print("\n❌ Validation failed! Fix implementation before benchmarking.")
            return
        else:
            print("\n✅ All validation checks passed!")

    # State operations
    if args.state or args.all:
        print("\nBenchmarking state operations...")
        results = Benchmark.benchmark_state_operations()
        print(format_results(results, "State Operation Benchmarks (ms)"))

    # Neural network
    if args.network or args.all:
        print("\nBenchmarking neural network...")
        results = Benchmark.benchmark_neural_network()
        print(format_results(results, "Neural Network Benchmarks"))

    # CFR iterations
    if args.cfr or args.all:
        print("\nBenchmarking CFR iterations...")
        config = get_quick_config()
        config["traversals_per_iteration"] = 100
        results = Benchmark.benchmark_cfr_iteration(config, num_iterations=5)
        print(format_results(results, "CFR Iteration Benchmarks"))

    # Game statistics
    if args.games or args.all:
        print("\nBenchmarking game statistics...")
        results = Benchmark.benchmark_game_length(num_games=50)
        print(format_results(results, "Game Length Statistics"))

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
