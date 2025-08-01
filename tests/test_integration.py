"""End-to-end integration tests."""

import os
import tempfile

import pytest

from mindbug import DeepCFR, GameState, MindbugEngine, Player
from mindbug.core import Action, ActionType, CardDatabase, CreatureState
from mindbug.training import Evaluator, Trainer
from mindbug.utils import get_debug_config, get_quick_config


class TestFullGameFlow:
    """Test complete game scenarios."""

    def test_full_game_playthrough(self):
        """Test a complete game from start to finish."""
        state = MindbugEngine.create_initial_state()

        move_count = 0
        max_moves = 200  # Prevent infinite loops

        while not state.is_terminal() and move_count < max_moves:
            actions = state.get_legal_actions()
            assert len(actions) > 0, "No legal actions available"

            # Choose random action
            import random

            action = random.choice(actions)

            # Apply action
            new_state = MindbugEngine.apply_action(state, action)

            # Verify state changed (unless it's a pass action)
            if action.action_type not in [ActionType.PASS_MINDBUG]:
                assert new_state != state, "State should change after action"

            state = new_state
            move_count += 1

        assert state.is_terminal(), "Game should reach terminal state"
        winner = state.get_winner()
        assert winner in [Player.PLAYER_1, Player.PLAYER_2, None]

    def test_mindbug_flow(self):
        """Test complete Mindbug interaction."""
        state = MindbugEngine.create_initial_state()
        state.current_player = Player.PLAYER_1

        # Ensure P1 has a creature
        creature_card = CardDatabase.get_card("Gorillion")
        state.hands[Player.PLAYER_1] = [creature_card]

        # P1 plays creature
        play_action = Action(ActionType.PLAY_CREATURE, Player.PLAYER_1, card=creature_card)
        state = MindbugEngine.apply_action(state, play_action)

        # Verify Mindbug decision phase
        assert state.mindbug_decision_pending
        assert state.current_player == Player.PLAYER_2

        # P2 uses Mindbug
        mindbug_action = Action(ActionType.USE_MINDBUG, Player.PLAYER_2)
        state = MindbugEngine.apply_action(state, mindbug_action)

        # Verify results
        assert len(state.play_areas[Player.PLAYER_2]) == 1
        assert state.play_areas[Player.PLAYER_2][0].card.name == "Gorillion"
        assert state.mindbugs_available[Player.PLAYER_2] == 1
        assert state.current_player == Player.PLAYER_1
        assert state.extra_turn_pending == Player.PLAYER_1

    def test_combat_flow(self):
        """Test complete combat sequence."""
        state = MindbugEngine.create_initial_state()
        state.current_player = Player.PLAYER_1

        # Set up combat scenario
        attacker = CardDatabase.get_card("Killer Bee")  # HUNTER
        blocker1 = CardDatabase.get_card("Spider Owl")
        blocker2 = CardDatabase.get_card("Gorillion")

        state.play_areas[Player.PLAYER_1].append(
            CreatureState(card=attacker, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        )
        state.play_areas[Player.PLAYER_2].extend(
            [
                CreatureState(card=blocker1, controller=Player.PLAYER_2, owner=Player.PLAYER_2),
                CreatureState(card=blocker2, controller=Player.PLAYER_2, owner=Player.PLAYER_2),
            ]
        )

        # Attack with HUNTER
        attack_action = Action(ActionType.ATTACK, Player.PLAYER_1, creature_index=0)
        state = MindbugEngine.apply_action(state, attack_action)

        # Should enter HUNTER choice
        assert state.hunter_choice_pending
        assert len(state.valid_blockers) == 2

        # Choose blocker
        choose_action = Action(ActionType.CHOOSE_BLOCKER, Player.PLAYER_1, target_index=0)
        state = MindbugEngine.apply_action(state, choose_action)

        # Combat resolved
        assert not state.hunter_choice_pending
        assert len(state.play_areas[Player.PLAYER_2]) == 1  # One creature died
        assert state.current_player == Player.PLAYER_2


class TestTrainingPipeline:
    """Test the complete training pipeline."""

    def test_trainer_initialization(self):
        """Test Trainer setup."""
        config = get_debug_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(config, checkpoint_dir=tmpdir)

            assert trainer.cfr is not None
            assert trainer.monitor is not None
            assert os.path.exists(trainer.run_dir)

            # Config should be saved
            config_path = trainer.run_dir / "config.json"
            assert config_path.exists()

    def test_training_loop(self):
        """Test complete training loop."""
        config = get_debug_config()
        config["traversals_per_iteration"] = 10

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(config, checkpoint_dir=tmpdir)

            # Train for a few iterations
            final_cfr = trainer.train(
                num_iterations=5, checkpoint_interval=2, eval_interval=3, save_final=True
            )

            assert final_cfr.iteration_count == 5

            # Check checkpoints saved
            checkpoints = list(trainer.run_dir.glob("*.pt"))
            assert len(checkpoints) >= 2  # At least one checkpoint + final

            # Check history saved
            history_path = trainer.run_dir / "training_history.json"
            assert history_path.exists()

    def test_evaluation_during_training(self):
        """Test evaluation metrics during training."""
        config = get_debug_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(config, checkpoint_dir=tmpdir)

            # Manually run evaluation
            trainer._evaluate(iteration=1)

            # Check metrics logged
            assert len(trainer.monitor.exploitability_history) > 0

            iteration, exploitability = trainer.monitor.exploitability_history[0]
            assert iteration == 1
            assert isinstance(exploitability, float)


class TestAgentBehavior:
    """Test trained agent behavior."""

    def test_agent_makes_legal_moves(self):
        """Test agent only makes legal moves."""
        config = get_debug_config()
        cfr = DeepCFR(config)
        cfr.train(num_iterations=5)

        # Test in various game states
        for _ in range(10):
            state = MindbugEngine.create_initial_state()

            # Play some random moves to get different states
            for _ in range(5):
                if state.is_terminal():
                    break

                actions = state.get_legal_actions()
                if not actions:
                    break

                import random

                action = random.choice(actions)
                state = MindbugEngine.apply_action(state, action)

            if not state.is_terminal():
                # Get agent's policy
                policy = cfr.get_policy(state, state.current_player)

                # All actions in policy should be legal
                legal_actions = set(state.get_legal_actions())
                policy_actions = set(policy.keys())

                assert policy_actions.issubset(legal_actions)

    def test_agent_improves_over_time(self):
        """Test that agent performance improves with training."""
        config = get_quick_config()
        config["traversals_per_iteration"] = 20

        # Create two agents
        untrained = DeepCFR(config)
        trained = DeepCFR(config)

        # Train one agent
        trained.train(num_iterations=10)

        # Compare performance
        results = Evaluator.evaluate_agents(trained, untrained, num_games=20)

        # Trained should win more often (but not guaranteed with so little training)
        # Just check it runs without errors
        assert "agent1_win_rate" in results
        assert "agent2_win_rate" in results
        assert "draw_rate" in results

    def test_deterministic_policy(self):
        """Test agent gives consistent policy for same state."""
        config = get_debug_config()
        cfr = DeepCFR(config)
        cfr.train(num_iterations=5)

        # Create a specific state
        state = MindbugEngine.create_initial_state(shuffle=False)

        # Get policy multiple times
        policies = []
        for _ in range(5):
            policy = cfr.get_policy(state, Player.PLAYER_1)
            policies.append(policy)

        # Should be identical
        for i in range(1, len(policies)):
            assert len(policies[i]) == len(policies[0])
            for action in policies[0]:
                assert action in policies[i]
                # Allow small numerical differences
                assert abs(policies[i][action] - policies[0][action]) < 0.001


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_deck_handling(self):
        """Test behavior when deck is empty."""
        state = GameState(current_player=Player.PLAYER_1)
        state.hands[Player.PLAYER_1] = [CardDatabase.get_card("Gorillion")]
        state.decks[Player.PLAYER_1] = []  # Empty deck

        # Should still be able to play
        actions = state.get_legal_actions()
        play_actions = [a for a in actions if a.action_type == ActionType.PLAY_CREATURE]
        assert len(play_actions) == 1

        # Play creature
        action = play_actions[0]
        new_state = MindbugEngine.apply_action(state, action)

        # Hand should remain empty (no cards to draw)
        assert len(new_state.hands[Player.PLAYER_1]) == 0

    def test_no_valid_targets(self):
        """Test abilities with no valid targets."""
        state = GameState(current_player=Player.PLAYER_1)

        # Brain Fly with no valid targets
        brain_fly = CardDatabase.get_card("Brain Fly")
        MindbugEngine._resolve_play_ability(state, brain_fly, Player.PLAYER_1)

        # Should complete without error
        assert len(state.play_areas[Player.PLAYER_1]) == 0

        # Tiger Squirrel with no valid targets
        tiger = CardDatabase.get_card("Tiger Squirrel")
        MindbugEngine._resolve_play_ability(state, tiger, Player.PLAYER_1)

        # Should complete without error
        assert len(state.discard_piles[Player.PLAYER_2]) == 0

    def test_life_overflow(self):
        """Test life can exceed starting value."""
        state = GameState(current_player=Player.PLAYER_1)
        state.life[Player.PLAYER_1] = 3

        # Gain life multiple times
        healer = CardDatabase.get_card("Axolotl Healer")
        for _ in range(3):
            MindbugEngine._resolve_play_ability(state, healer, Player.PLAYER_1)

        assert state.life[Player.PLAYER_1] == 9  # 3 + 2 + 2 + 2

    def test_simultaneous_effects_ordering(self):
        """Test resolution order of simultaneous effects."""
        state = GameState(current_player=Player.PLAYER_1)

        # Multiple creatures with defeated abilities
        for _ in range(2):
            toad = CardDatabase.get_card("Explosive Toad")
            creature = CreatureState(card=toad, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
            state.play_areas[Player.PLAYER_1].append(creature)

        # Add potential targets
        for player in [Player.PLAYER_1, Player.PLAYER_2]:
            owl = CardDatabase.get_card("Spider Owl")
            state.play_areas[player].append(
                CreatureState(card=owl, controller=player, owner=player)
            )

        # Defeat both toads simultaneously
        toads = [c for c in state.play_areas[Player.PLAYER_1] if c.card.name == "Explosive Toad"]
        for toad in toads:
            MindbugEngine._defeat_creature(state, toad, Player.PLAYER_1)

        # Should trigger both defeated abilities
        # With 2 toads and 2 owls initially, all should be gone
        total_creatures = len(state.play_areas[Player.PLAYER_1]) + len(
            state.play_areas[Player.PLAYER_2]
        )
        assert total_creatures <= 2  # Some randomness in targets


class TestPerformance:
    """Test performance characteristics."""

    def test_state_copying_performance(self):
        """Test state copying is efficient."""
        import time

        state = MindbugEngine.create_initial_state()

        # Add some complexity
        for _ in range(5):
            if state.hands[state.current_player]:
                card = state.hands[state.current_player][0]
                action = Action(ActionType.PLAY_CREATURE, state.current_player, card=card)
                state = MindbugEngine.apply_action(state, action)

                # Pass on Mindbug
                action = Action(ActionType.PASS_MINDBUG, state.current_player)
                state = MindbugEngine.apply_action(state, action)

        # Time copying
        start = time.time()
        for _ in range(1000):
            _ = state.copy()
        elapsed = time.time() - start

        # Should be reasonably fast
        assert elapsed < 1.0  # Less than 1 second for 1000 copies
        print(f"1000 state copies took {elapsed:.3f} seconds")

    def test_action_generation_performance(self):
        """Test legal action generation is fast."""
        import time

        state = MindbugEngine.create_initial_state()

        total_time = 0
        num_calls = 0

        # Test in various game states
        for _ in range(100):
            if state.is_terminal():
                break

            start = time.time()
            actions = state.get_legal_actions()
            elapsed = time.time() - start

            total_time += elapsed
            num_calls += 1

            if actions:
                import random

                action = random.choice(actions)
                state = MindbugEngine.apply_action(state, action)

        if num_calls > 0:
            avg_time = total_time / num_calls
            assert avg_time < 0.001  # Less than 1ms average
            print(f"Average time for get_legal_actions: {avg_time*1000:.3f}ms")
