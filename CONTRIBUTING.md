# Contributing to Mindbug Deep CFR

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub flow, so all code changes happen through pull requests:

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code follows the style guidelines
6. Issue that pull request!

## Setting Up Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/mindbug-deep-cfr.git
cd mindbug-deep-cfr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
make install-dev

# Verify setup
make test-fast
```

## Code Style

We use several tools to maintain code quality:

```bash
# Format code automatically
make format

# Check code style
make lint

# Type checking
make typecheck

# Run all checks
make dev
```

### Style Guidelines

- **Black** for code formatting (100 char line length)
- **isort** for import sorting
- **ruff** for linting
- **mypy** for type checking
- Type hints for all functions
- Docstrings for all public functions/classes

## Testing

### Running Tests

```bash
# Fast tests only
make test

# All tests
make test-all

# Specific test file
pytest tests/test_cards.py -v

# With coverage
make coverage
```

### Writing Tests

- Test files go in `tests/`
- Name test files `test_*.py`
- Use descriptive test names that explain what's being tested
- Aim for >80% code coverage
- Test edge cases and error conditions

Example test:

```python
def test_brain_fly_steals_high_power_creature(self):
    """Test Brain Fly takes control of 6+ power creature."""
    state = GameState(current_player=Player.PLAYER_1)
    
    # Setup
    target = CardDatabase.get_card("Gorillion")  # 10 power
    creature = CreatureState(card=target, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
    state.play_areas[Player.PLAYER_2].append(creature)
    
    # Execute
    brain_fly = CardDatabase.get_card("Brain Fly")
    MindbugEngine._resolve_play_ability(state, brain_fly, Player.PLAYER_1)
    
    # Assert
    assert len(state.play_areas[Player.PLAYER_1]) == 1
    assert state.play_areas[Player.PLAYER_1][0].card.name == "Gorillion"
```

## Adding New Features

### 1. Card Implementations

When adding new cards:

1. Add card definition to `mindbug/core/cards.py`
2. Implement ability in `mindbug/core/engine.py`
3. Add comprehensive tests in `tests/test_cards.py`
4. Update documentation

### 2. Algorithm Improvements

When modifying Deep CFR:

1. Ensure mathematical correctness
2. Add unit tests for new components
3. Run convergence tests
4. Document any deviations from the paper

### 3. Performance Optimizations

When optimizing:

1. Benchmark before and after
2. Document the optimization
3. Ensure tests still pass
4. Consider GPU memory usage

## Documentation

- Update docstrings for any changed functions
- Update README.md if adding major features
- Include type hints
- Add examples for complex functionality

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the tests to cover your changes
3. Ensure all tests pass locally
4. Update documentation as needed
5. The PR will be merged once you have approval from a maintainer

### PR Title Format

Use conventional commit format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only changes
- `style:` Code style changes (formatting, etc)
- `refactor:` Code change that neither fixes a bug nor adds a feature
- `perf:` Performance improvement
- `test:` Adding missing tests
- `chore:` Changes to build process or auxiliary tools

Example: `feat: add support for custom card sets`

## Reporting Bugs

### Security Vulnerabilities

If you find a security vulnerability, do NOT open an issue. Email security@mindbug-ai.example.com instead.

### Bug Reports

When filing an issue, make sure to answer these questions:

1. What version are you using?
2. What environment (OS, Python version, GPU)?
3. What did you do?
4. What did you expect to see?
5. What did you see instead?

Include:
- Steps to reproduce
- Error messages
- Minimal code example

## Feature Requests

We love feature requests! But please:

1. Check if it's already requested
2. Clearly describe the feature
3. Explain why it's needed
4. Give examples of how it would be used

## Community

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## Questions?

Feel free to open an issue with the `question` label or reach out on our Discord server.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in:
- The README.md file
- Release notes
- Project documentation

Thank you for contributing to Mindbug Deep CFR! 🎮🐛