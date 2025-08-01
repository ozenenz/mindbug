#!/usr/bin/env python3
"""Run comprehensive test suite with reporting."""
import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_tests(args):
    """Run test suite with specified options."""
    cmd = ["python", "-m", "pytest", "tests/", "-v"]

    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=mindbug", "--cov-report=term-missing", "--cov-report=html"])

    # Add markers
    markers = []
    if args.fast:
        markers.append("not slow")
    if not args.gpu:
        markers.append("not gpu")

    if markers:
        cmd.extend(["-m", " and ".join(markers)])

    # Add specific test file if provided
    if args.test:
        cmd = ["python", "-m", "pytest", f"tests/{args.test}", "-v"]

    # Add pytest options
    if args.verbose:
        cmd.append("-vv")
    if args.exitfirst:
        cmd.append("-x")
    if args.pdb:
        cmd.append("--pdb")

    # Run tests
    print(f"Running: {' '.join(cmd)}")
    print("=" * 70)

    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time

    print("=" * 70)
    print(f"Tests completed in {elapsed:.2f} seconds")

    return result.returncode


def run_specific_test_suites():
    """Run specific test categories with summaries."""
    suites = {
        "cards": "test_cards.py",
        "engine": "test_engine.py",
        "cfr": "test_cfr.py",
        "integration": "test_integration.py",
    }

    results = {}

    for name, file in suites.items():
        print(f"\n{'='*70}")
        print(f"Running {name.upper()} tests...")
        print(f"{'='*70}")

        cmd = ["python", "-m", "pytest", f"tests/{file}", "-v", "--tb=short"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse results
        output = result.stdout
        if "passed" in output:
            # Extract test counts
            import re

            match = re.search(r"(\d+) passed", output)
            passed = int(match.group(1)) if match else 0
            results[name] = {"passed": passed, "failed": 0}
        else:
            results[name] = {"passed": 0, "failed": 1}

        if result.returncode != 0:
            print(f"❌ {name} tests FAILED")
            print(result.stdout[-500:])  # Last 500 chars
        else:
            print(f"✅ {name} tests PASSED")

    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")

    total_passed = sum(r["passed"] for r in results.values())
    total_failed = sum(r["failed"] for r in results.values())

    for name, result in results.items():
        status = "✅ PASS" if result["failed"] == 0 else "❌ FAIL"
        print(f"{name.ljust(15)} {status} ({result['passed']} passed)")

    print(f"\nTotal: {total_passed} passed, {total_failed} failed")

    return 0 if total_failed == 0 else 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Mindbug Deep CFR test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python run_tests.py
  
  # Run with coverage
  python run_tests.py --coverage
  
  # Run fast tests only
  python run_tests.py --fast
  
  # Run specific test file
  python run_tests.py --test test_cards.py
  
  # Run test suites with summary
  python run_tests.py --suites
        """,
    )

    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--gpu", action="store_true", help="Include GPU tests")
    parser.add_argument("--test", help="Run specific test file")
    parser.add_argument("--verbose", action="store_true", help="Extra verbose output")
    parser.add_argument("--exitfirst", action="store_true", help="Exit on first failure")
    parser.add_argument("--pdb", action="store_true", help="Drop into debugger on failures")
    parser.add_argument("--suites", action="store_true", help="Run test suites with summary")

    args = parser.parse_args()

    # Ensure we're in the right directory
    if not Path("tests").exists():
        print("Error: tests/ directory not found. Run from project root.")
        sys.exit(1)

    # Run tests
    if args.suites:
        return_code = run_specific_test_suites()
    else:
        return_code = run_tests(args)

    sys.exit(return_code)


if __name__ == "__main__":
    main()
