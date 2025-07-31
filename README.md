# Mindbug First Contact - Deep CFR Implementation

A state-of-the-art implementation of Mindbug First Contact with Deep Counterfactual Regret Minimization (Deep CFR) for learning optimal strategies.

## 🎮 About Mindbug

Mindbug is a tactical card game where players summon creatures to battle, but with a twist - your opponent can use "Mindbugs" to take control of your creatures as you play them. This creates a unique dynamic of strategic decision-making and mind games.

This implementation includes all 32 cards from the First Contact set with their complete abilities and interactions.

## 🤖 About Deep CFR

Deep Counterfactual Regret Minimization is a scalable algorithm for finding Nash equilibrium strategies in large imperfect information games. This implementation follows the algorithm exactly as specified in the research paper, with proper external sampling Monte Carlo and neural network approximation.

## 📋 Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- 16GB+ RAM
- 10GB+ free disk space for checkpoints

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mindbug-deep-cfr.git
cd mindbug-deep-cfr

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
python -m mindbug.test_framework
```

### Basic Training

```bash
# Quick test run (100 iterations, ~5 minutes)
python train_improved.py --config quick --iterations 100

# Full training run (10,000 iterations, ~12 hours on good GPU)
python train_improved.py --config performance --iterations 10000

# Monitor training progress
tensorboard --logdir checkpoints/
```

### Playing Against Trained Agent

```python
# Interactive play against AI
python play.py

# Or programmatically:
from mindbug.algorithms import DeepCFR
from mindbug.game import MindbugEngine, Player
from play import play_interactive_game

# Load trained model
cfr = DeepCFR({"use_gpu": False})
cfr.load_checkpoint("checkpoints/final_checkpoint.pt")

# Play interactive game
play_interactive_game(cfr, human_player=Player.PLAYER_1)
```

## 🏗️ Architecture

### Game Engine (`mindbug/game/`)
- `engine.py` - Core game logic and rules enforcement
- `state.py` - Game state representation
- `cards.py` - All 32 card definitions
- `constants.py` - Game constants and enums
- `actions.py` - Action representation
- `state_pool.py` - Optimized state management

### Deep CFR Algorithm (`mindbug/algorithms/`)
- `deep_cfr.py` - Main Deep CFR implementation
- `networks.py` - Dual-branch neural network architecture

### Training & Evaluation (`mindbug/training/`)
- `train_improved.py` - Advanced training script with monitoring
- `evaluate.py` - Agent evaluation utilities
- `test_framework.py` - Comprehensive test suite

### Configuration (`mindbug/utils/`)
- `config.py` - Preset configurations for different scenarios

## ⚙️ Configuration Options

### Quick Test Config
- Fast iteration for debugging
- 100 traversals per iteration
- Small buffers (50k samples)
- Reduced network size

### Performance Config
- Production settings
- 1000 traversals per iteration
- Large buffers (2M samples)
- Full network architecture

### Distributed Config
- Multi-GPU training
- 5000 traversals per iteration
- Huge buffers (10M samples)
- Optimized batch sizes

### Debug Config
- CPU-only execution
- Minimal settings
- Verbose logging
- Small batch sizes

## 📊 Monitoring Training

Training progress can be monitored in real-time using TensorBoard:

```bash
tensorboard --logdir checkpoints/your_run_name/tensorboard/
```

Key metrics to watch:
- **Exploitability** - Should decrease toward 0
- **Win rates** - Should balance around 50/50
- **Buffer sizes** - Should grow steadily
- **GPU memory** - Should remain stable

## 🧪 Testing

### Run All Tests
```bash
python -m mindbug.test_framework
```

### Run Specific Test Categories
```python
from mindbug.test_framework import TestCardImplementations, TestCombatResolution

# Test card implementations
test = TestCardImplementations()
test.test_sharky_crab_dog_mummypus()
test.test_kangasaurus_rex_mass_removal()

# Test combat mechanics
combat = TestCombatResolution()
combat.test_tough_vs_poisonous()
```

## 🎯 Training Tips

1. **Start with validation** - Always run tests before training to ensure correctness
2. **Use appropriate config** - Match config to your hardware capabilities
3. **Monitor convergence** - Watch exploitability decrease over time
4. **Save checkpoints** - Training can be resumed from any checkpoint
5. **Batch size matters** - Larger batches = faster training (if GPU memory allows)

## 📈 Expected Results

With proper training:
- **Exploitability < 0.05** within 5,000 iterations
- **Exploitability < 0.01** within 20,000 iterations
- **Near-optimal play** within 50,000 iterations

Training time depends on hardware:
- **RTX 3090**: ~50 iterations/second
- **RTX 4090**: ~150 iterations/second
- **A100**: ~200 iterations/second

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch_size in config
- Use gradient accumulation
- Try mixed precision training

### Slow Training
- Ensure GPU is being used
- Check batch size is appropriate
- Verify no CPU bottlenecks

### Convergence Issues
- Verify tests pass
- Check learning rates
- Ensure proper CFR implementation

## 📚 References

- [Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164) - Brown et al., 2019
- [Mindbug Official Rules](https://mindbug.me/rules)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 📄 License

This implementation is for educational and research purposes. Mindbug is a trademark of Nerdlab Games.

## 🤝 Contributing

Contributions are welcome! Please ensure:
1. All tests pass
2. Code follows existing style
3. New features include tests
4. Documentation is updated

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Happy Training! May your Mindbugs be ever in your favor! 🐛🎮**