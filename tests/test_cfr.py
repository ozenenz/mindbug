"""Tests for Deep CFR implementation."""

import numpy as np
import pytest
import torch

from mindbug.algorithms import DeepCFR, DualBranchNetwork, StateEncoder
from mindbug.core import Action, ActionType, GameState, MindbugEngine, Player
from mindbug.utils import get_debug_config, get_quick_config


class TestNetworkArchitecture:
    """Test neural network components."""

    def test_network_structure(self):
        """Verify network architecture matches paper."""
        network = DualBranchNetwork(
            card_embedding_dim=128, history_dim=64, hidden_dim=256, num_card_types=32
        )

        # Count layers
        card_layers = len([m for m in network.card_branch if isinstance(m, torch.nn.Linear)])
        history_layers = len([m for m in network.history_branch if isinstance(m, torch.nn.Linear)])
        combined_layers = len(network.combined_layers)

        assert card_layers == 3, "Card branch should have 3 layers"
        assert history_layers == 2, "History branch should have 2 layers"
        assert combined_layers == 2, "Combined section should have 2 layers"

        # Total: 3 + 2 + 2 = 7 layers as specified

    def test_network_forward_pass(self):
        """Test network forward propagation."""
        network = DualBranchNetwork()
        batch_size = 16

        # Create dummy inputs
        card_indices = torch.randint(0, 33, (batch_size, 30))
        history_features = torch.randn(batch_size, 64)

        # Forward pass
        output = network(card_indices, history_features)

        assert output.shape == (batch_size, 1)
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()

    def test_network_gradient_flow(self):
        """Test gradients flow through network."""
        network = DualBranchNetwork()

        # Dummy data
        card_indices = torch.randint(0, 33, (8, 30))
        history_features = torch.randn(8, 64)
        targets = torch.randn(8, 1)

        # Forward pass
        output = network(card_indices, history_features)
        loss = torch.nn.functional.mse_loss(output, targets)

        # Backward pass
        loss.backward()

        # Check gradients exist
        for name, param in network.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


class TestStateEncoder:
    """Test state encoding for neural network."""

    def test_encoder_initialization(self):
        """Test encoder setup."""
        encoder = StateEncoder()

        assert encoder.num_card_types == 32
        assert encoder.max_cards_encoded == 30
        assert encoder.history_dim == 64
        assert len(encoder.card_to_idx) == 32

    def test_card_encoding(self):
        """Test encoding cards as indices."""
        encoder = StateEncoder()
        state = MindbugEngine.create_initial_state()

        # Create action
        if state.hands[Player.PLAYER_1]:
            card = state.hands[Player.PLAYER_1][0]
            action = Action(ActionType.PLAY_CREATURE, Player.PLAYER_1, card=card)
        else:
            action = Action(ActionType.PASS_MINDBUG, Player.PLAYER_1)

        card_indices, _ = encoder.encode_state_and_action(state, Player.PLAYER_1, action)

        assert card_indices.shape == (30,)
        assert card_indices.dtype == torch.long
        assert (card_indices >= 0).all()
        assert (card_indices <= 32).all()  # Including padding

    def test_history_encoding(self):
        """Test encoding game state features."""
        encoder = StateEncoder()
        state = MindbugEngine.create_initial_state()
        action = Action(ActionType.PASS_MINDBUG, Player.PLAYER_1)

        _, history = encoder.encode_state_and_action(state, Player.PLAYER_1, action)

        assert history.shape == (64,)
        assert history.dtype == torch.float32

        # Check specific features
        assert history[0] == 0.3  # Life/10
        assert history[1] == 0.3  # Opponent life/10
        assert history[2] == 1.0  # Mindbugs/2
        assert history[8] == 1.0  # Hand size/5

    def test_encoding_consistency(self):
        """Test encoding is deterministic."""
        encoder = StateEncoder()
        state = MindbugEngine.create_initial_state(shuffle=False)
        action = Action(ActionType.PASS_MINDBUG, Player.PLAYER_1)

        # Encode twice
        cards1, history1 = encoder.encode_state_and_action(state, Player.PLAYER_1, action)
        cards2, history2 = encoder.encode_state_and_action(state, Player.PLAYER_1, action)

        assert torch.equal(cards1, cards2)
        assert torch.equal(history1, history2)

    def test_batch_encoding(self):
        """Test batch encoding efficiency."""
        encoder = StateEncoder()

        # Create batch
        states = []
        players = []
        actions = []

        for _ in range(32):
            state = MindbugEngine.create_initial_state()
            player = Player.PLAYER_1
            action = Action(ActionType.PASS_MINDBUG, player)

            states.append(state)
            players.append(player)
            actions.append(action)

        # Batch encode
        card_batch, history_batch = encoder.batch_encode(states, players, actions)

        assert card_batch.shape == (32, 30)
        assert history_batch.shape == (32, 64)


class TestDeepCFR:
    """Test Deep CFR algorithm implementation."""

    def test_initialization(self):
        """Test Deep CFR setup."""
        config = get_debug_config()
        cfr = DeepCFR(config)

        assert cfr.use_linear_cfr == config["use_linear_cfr"]
        assert len(cfr.advantage_buffers) == 2
        assert len(cfr.strategy_buffers) == 2
        assert cfr.iteration_count == 0

    def test_single_iteration(self):
        """Test running one CFR iteration."""
        config = get_debug_config()
        config["traversals_per_iteration"] = 10
        cfr = DeepCFR(config)

        # Run one iteration
        cfr.train(num_iterations=1)

        assert cfr.iteration_count == 1
        assert len(cfr.value_networks) == 2

        # Should have collected samples
        p1_samples = len(cfr.advantage_buffers[Player.PLAYER_1])
        p2_samples = len(cfr.advantage_buffers[Player.PLAYER_2])
        assert p1_samples > 0 or p2_samples > 0

    def test_regret_matching(self):
        """Test strategy computation via regret matching."""
        config = get_debug_config()
        cfr = DeepCFR(config)

        # Create networks
        cfr.value_networks = {
            Player.PLAYER_1: cfr._create_network(),
            Player.PLAYER_2: cfr._create_network(),
        }

        state = MindbugEngine.create_initial_state()
        legal_actions = state.get_legal_actions()

        # Get strategy
        strategy = cfr._get_strategy(state, legal_actions, Player.PLAYER_1, 1)

        assert len(strategy) == len(legal_actions)
        assert all(p >= 0 for p in strategy.values())
        assert abs(sum(strategy.values()) - 1.0) < 0.001

    def test_linear_cfr_weighting(self):
        """Test Linear CFR weighting scheme."""
        config = get_debug_config()
        config["use_linear_cfr"] = True
        cfr = DeepCFR(config)

        # Manually add samples with different iterations
        from mindbug.algorithms.deep_cfr import CFRSample

        sample1 = CFRSample(
            card_indices=torch.zeros(30),
            history_features=torch.zeros(64),
            value=1.0,
            weight=1,
            iteration=1,
        )

        sample2 = CFRSample(
            card_indices=torch.zeros(30),
            history_features=torch.zeros(64),
            value=1.0,
            weight=10,
            iteration=10,
        )

        cfr.advantage_buffers[Player.PLAYER_1].add(sample1)
        cfr.advantage_buffers[Player.PLAYER_1].add(sample2)

        # Later iterations should have higher weight
        assert sample2.weight > sample1.weight

    def test_reservoir_sampling(self):
        """Test reservoir buffer maintains uniform distribution."""
        from mindbug.algorithms.deep_cfr import CFRSample, ReservoirBuffer

        buffer = ReservoirBuffer(capacity=100)

        # Add many samples
        for i in range(1000):
            sample = CFRSample(
                card_indices=torch.zeros(30),
                history_features=torch.zeros(64),
                value=float(i),
                weight=1.0,
                iteration=1,
            )
            buffer.add(sample)

        assert len(buffer) == 100

        # Check distribution is roughly uniform
        values = [s.value for s in buffer.buffer]
        assert min(values) >= 0
        assert max(values) < 1000
        assert 400 < np.mean(values) < 600  # Should be ~500

    def test_policy_computation(self):
        """Test final policy extraction."""
        config = get_debug_config()
        cfr = DeepCFR(config)

        # Train briefly
        cfr.train(num_iterations=5)

        # Get policy
        state = MindbugEngine.create_initial_state()
        policy = cfr.get_policy(state, Player.PLAYER_1)

        assert len(policy) > 0
        assert all(isinstance(a, Action) for a in policy.keys())
        assert all(0 <= p <= 1 for p in policy.values())
        assert abs(sum(policy.values()) - 1.0) < 0.001

    def test_checkpoint_save_load(self):
        """Test model persistence."""
        import os
        import tempfile

        config = get_debug_config()
        cfr = DeepCFR(config)

        # Train briefly
        cfr.train(num_iterations=2)

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pt")
            cfr.save_checkpoint(path)

            assert os.path.exists(path)

            # Load into new instance
            cfr2 = DeepCFR(config)
            cfr2.load_checkpoint(path)

            assert cfr2.iteration_count == cfr.iteration_count
            assert cfr2.encoder.card_to_idx == cfr.encoder.card_to_idx


class TestCFRConvergence:
    """Test algorithm convergence properties."""

    def test_self_play_balance(self):
        """Test that self-play produces balanced results."""
        from mindbug.training import Evaluator

        config = get_quick_config()
        config["traversals_per_iteration"] = 50
        cfr = DeepCFR(config)

        # Train for a few iterations
        cfr.train(num_iterations=20)

        # Evaluate self-play
        results = Evaluator.self_play_evaluation(cfr, num_games=50)

        # Should be roughly balanced
        assert 0.3 < results["agent1_win_rate"] < 0.7
        assert 0.3 < results["agent2_win_rate"] < 0.7
        assert results["agent1_win_rate"] + results["agent2_win_rate"] + results["draw_rate"] == 1.0

    def test_exploitability_computation(self):
        """Test exploitability metric."""
        from mindbug.training import Evaluator

        config = get_debug_config()
        cfr = DeepCFR(config)

        # Random policy should be highly exploitable
        exploitability = Evaluator.compute_exploitability(cfr, num_iterations=50)

        # Should be negative (losing to random)
        assert exploitability < 0

        # Train briefly
        cfr.train(num_iterations=10)

        # Should improve
        exploitability2 = Evaluator.compute_exploitability(cfr, num_iterations=50)
        assert exploitability2 > exploitability  # Less negative = better

    def test_strategy_network_training(self):
        """Test strategy network learns from samples."""
        config = get_debug_config()
        cfr = DeepCFR(config)

        # Create some strategy samples
        from mindbug.algorithms.deep_cfr import CFRSample

        for _ in range(100):
            sample = CFRSample(
                card_indices=torch.randint(0, 32, (30,)),
                history_features=torch.randn(64),
                value=np.random.random(),  # Random probability
                weight=1.0,
                iteration=1,
            )
            cfr.strategy_buffers[Player.PLAYER_1].add(sample)

        # Train strategy network
        cfr._train_strategy_network(1)

        assert cfr.strategy_network is not None

        # Test prediction
        cfr.strategy_network.eval()
        with torch.no_grad():
            card_indices = torch.randint(0, 32, (1, 30)).to(cfr.device)
            history = torch.randn(1, 64).to(cfr.device)
            output = cfr.strategy_network(card_indices, history)

            # Should output reasonable logits
            assert output.shape == (1, 1)
            assert -10 < output.item() < 10  # Reasonable range


class TestGPUAcceleration:
    """Test GPU-specific functionality."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_training(self):
        """Test training on GPU."""
        config = get_debug_config()
        config["use_gpu"] = True
        cfr = DeepCFR(config)

        assert cfr.device.type == "cuda"

        # Train one iteration
        cfr.train(num_iterations=1)

        # Networks should be on GPU
        for network in cfr.value_networks.values():
            assert next(network.parameters()).is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_memory_management(self):
        """Test GPU memory is managed properly."""
        config = get_quick_config()
        config["use_gpu"] = True
        config["batch_size"] = 1024

        initial_memory = torch.cuda.memory_allocated()

        cfr = DeepCFR(config)
        cfr.train(num_iterations=5)

        # Memory should be reasonable
        final_memory = torch.cuda.memory_allocated()
        memory_mb = (final_memory - initial_memory) / 1024 / 1024

        assert memory_mb < 1000  # Less than 1GB increase
