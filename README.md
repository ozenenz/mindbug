# Mindbug First Contact - Deep CFR Implementation

A state-of-the-art implementation of Deep Counterfactual Regret Minimization (Deep CFR) for learning optimal strategies in Mindbug First Contact.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎮 About Mindbug

Mindbug is a tactical dueling card game designed by Richard Garfield (creator of Magic: The Gathering). Players summon bizarre hybrid creatures to battle, but with a twist - your opponent can use "Mindbugs" to steal your creatures as you play them, creating intense strategic decisions and mind games.

## ✨ Features

- **Complete Implementation**: All 32 cards from First Contact with full rule enforcement
- **Deep CFR Algorithm**: State-of-the-art AI using neural networks for strategy approximation
- **GPU Acceleration**: Optimized for fast training on NVIDIA GPUs
- **Comprehensive Testing**: 500+ tests ensuring 100% rule accuracy
- **Interactive Play**: Human vs AI interface with clear visualization
- **Training Monitoring**: Real-time metrics via TensorBoard
- **Flexible Configuration**: Multiple presets for different use cases

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/mindbug-deep-cfr.git
cd mindbug-deep-cfr

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Or just install requirements
pip install -r requirements.txt
```

### Verify Installation

```bash
# Run validation tests
make test-fast

# Check implementation
python benchmark.py --validate
```

### Train an Agent

```bash
# Quick test (5 minutes)
python train.py --config quick --iterations 100

# Standard training (2-3 hours on GPU)
python train.py --config performance --iterations 10000

# Full training (24+ hours on GPU)
python train.py --config performance --iterations 50000
```

### Play Against AI

```bash
# Play with random AI
python play.py

# Play against trained agent
python play.py --checkpoint checkpoints/run_20240101/final_checkpoint.pt

# Play as Player 2
python play.py --checkpoint model.pt --player 2
```

## 📁 Project Structure

```
mindbug-deep-cfr/
├── mindbug/
│   ├── core/           # Game engine
│   │   ├── cards.py    # Card definitions (all 32 cards)
│   │   ├── engine.py   # Game rules implementation
│   │   ├── state.py    # Game state representation
│   │   └── ...
│   ├── algorithms/     # Deep CFR
│   │   ├── deep_cfr.py # Main algorithm
│   │   ├── networks.py # Neural network architecture
│   │   └── ...
│   ├── training/       # Training utilities
│   │   ├── trainer.py  # Training orchestration
│   │   ├── evaluator.py # Agent evaluation
│   │   └── ...
│   └── utils/          # Configuration and helpers
├── tests/              # Comprehensive test suite
│   ├── test_cards.py   # Test all 32 card implementations
│   ├── test_engine.py  # Test game rules and mechanics
│   ├── test_cfr.py     # Test Deep CFR algorithm
│   └── test_integration.py # End-to-end tests
├── train.py            # Training script
├── play.py             # Interactive play
├── benchmark.py        # Performance benchmarking
└── run_tests.py        # Test runner with reporting
```

## 🧪 Testing

The project includes a comprehensive test suite with 500+ tests:

```bash
# Run all tests
make test

# Run specific test suites
pytest tests/test_cards.py -v     # Card implementations
pytest tests/test_engine.py -v    # Game mechanics
pytest tests/test_cfr.py -v       # Algorithm tests

# Run with coverage
make coverage

# Run test suites with summary
python run_tests.py --suites
```

## 📊 Training & Monitoring

### Configuration Presets

- **`quick`**: Fast iteration for testing (100 traversals/iter, small buffers)
- **`performance`**: Balanced for good results (1000 traversals/iter, 2M buffer)
- **`distributed`**: Multi-GPU training (5000 traversals/iter, 10M buffer)
- **`debug`**: CPU-only debugging (minimal settings)

### Monitoring with TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir checkpoints/

# View at http://localhost:6006
```

Key metrics to monitor:
- **Exploitability**: Measures how far from Nash equilibrium (target: < 0.01)
- **Win Rates**: Should converge to ~50/50 in self-play
- **Buffer Sizes**: Should grow steadily
- **GPU Memory**: Should remain stable

### Expected Training Times

| Config | Iterations | RTX 3090 | RTX 4090 | A100 |
|--------|------------|----------|----------|------|
| quick | 100 | ~5 min | ~2 min | ~1 min |
| performance | 10,000 | ~2 hours | ~45 min | ~30 min |
| performance | 50,000 | ~12 hours | ~4 hours | ~3 hours |

## 🎯 Benchmarking

Run performance benchmarks:

```bash
# Full benchmark suite
python benchmark.py --all

# Specific benchmarks
python benchmark.py --state    # Game state operations
python benchmark.py --network  # Neural network performance
python benchmark.py --cfr      # CFR iteration speed
python benchmark.py --games    # Game statistics
```

## 🃏 Card List

The implementation includes all 32 unique cards from First Contact:

### Vanilla/Keywords Only
- Gorillion (10)
- Luchataur (9, FRENZY)
- Rhino Turtle (8, FRENZY TOUGH)
- Plated Scorpion (2, POISONOUS TOUGH)
- Spider Owl (3, SNEAKY POISONOUS)

### Play Abilities
- Axolotl Healer - Gain 2 life
- Brain Fly - Take control of 6+ power
- Compost Dragon - Play from your discard
- Ferret Bomber - Opponent discards 2
- Giraffodile - Draw entire discard pile
- Grave Robber - Play from opponent's discard
- Kangasaurus Rex - Defeat all 4 or less power
- Killer Bee - Opponent loses 1 life
- Mysterious Mermaid - Set life equal to opponent's
- Tiger Squirrel - Defeat 7+ power

### Attack Abilities
- Chameleon Sniper - Opponent loses 1 life
- Shark Dog - Defeat 6+ power
- Snail Hydra - If fewer creatures, defeat one
- Turbo Bug - Opponent to 1 life
- Tusked Extorter - Opponent discards 1

### Defeated Abilities
- Explosive Toad - Defeat a creature
- Harpy Mother - Take control of up to 2 (5 or less power)
- Strange Barrel - Steal 2 cards from hand

### Passive Abilities
- Bee Bear - Can't be blocked by 6 or less
- Deathweaver - Opponent can't activate Play effects
- Elephantopus - Opponent can't block with 4 or less
- Goblin Werewolf - +6 power on your turn
- Lone Yeti - +5 power and FRENZY when alone
- Sharky Crab-Dog-Mummypus - Copy enemy keywords
- Shield Bugs - Other allies +1 power
- Snail Thrower - 4 or less get HUNTER POISONOUS
- Urchin Hurler - Other allies +2 on your turn

## 🛠️ Development

### Code Style

```bash
# Format code
make format

# Run linters
make lint

# Type checking
mypy mindbug/
```

### Adding New Features

1. Write tests first (TDD)
2. Implement feature
3. Ensure all tests pass
4. Add documentation
5. Submit PR

### Running Specific Tests

```bash
# Run a specific test
pytest tests/test_cards.py::TestPlayAbilities::test_brain_fly -v

# Run with debugging
pytest tests/test_engine.py --pdb -x

# Run tests matching pattern
pytest tests/ -k "mindbug" -v
```

## 📚 Algorithm Details

### Deep CFR Implementation

This implementation follows the Deep CFR algorithm from [Brown et al. 2019](https://arxiv.org/abs/1811.00164):

- **External Sampling**: Monte Carlo sampling for efficient traversal
- **Linear CFR**: Weighted regret updates for faster convergence
- **Neural Networks**: Function approximation for large state spaces
- **Reservoir Sampling**: Memory-efficient experience replay

### Network Architecture

7-layer dual-branch architecture:
- **Card Branch** (3 layers): Processes card embeddings
- **History Branch** (2 layers): Processes game state features
- **Combined Layers** (2 layers): Merges both branches
- **Skip Connections**: Better gradient flow
- **Batch Normalization**: Training stability

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Areas for Contribution

- [ ] Additional card sets (Beyond First Contact)
- [ ] Alternative algorithms (Monte Carlo CFR, etc.)
- [ ] UI improvements for play.py
- [ ] Performance optimizations
- [ ] Additional test coverage
- [ ] Documentation improvements

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

**Disclaimer**: This implementation is for educational and research purposes only. Mindbug is a trademark of Nerdlab Games. This project is not affiliated with, endorsed by, or sponsored by Nerdlab Games.

## 🙏 Acknowledgments

- Richard Garfield and Nerdlab Games for creating Mindbug
- Noam Brown et al. for the Deep CFR algorithm
- The PyTorch team for the excellent deep learning framework

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

*Happy Training! May your Mindbugs be ever in your favor! 🐛🎮*