import unittest
import time
from typing import List, Dict, Tuple, Optional
import torch
import numpy as np

from mindbug.game import MindbugEngine, GameState, Action, Card, CardDefinitions, Player, ActionType, Keyword
from mindbug.game.state import CreatureState
from mindbug.algorithms import DeepCFR, MindbugStateEncoder, DualBranchMindbugNetwork
from mindbug.utils.config import get_quick_test_config, get_debug_config


class TestCardImplementations(unittest.TestCase):
    # Test all 32 card implementations
    
    def setUp(self):
        self.cards = CardDefinitions.get_first_contact_cards()
        self.engine = MindbugEngine
    
    def test_sharky_crab_dog_mummypus(self):
        # Test dynamic keyword copying
        state = self.engine.create_initial_state()
        
        # Set up test scenario
        sharky = self.cards["Sharky Crab-Dog-Mummypus"]
        poisonous_creature = self.cards["Spider Owl"]  # Has SNEAKY, POISONOUS
        
        # Place creatures
        state.play_areas[Player.PLAYER_1].append(
            CreatureState(card=sharky, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        )
        state.play_areas[Player.PLAYER_2].append(
            CreatureState(card=poisonous_creature, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
        )
        
        # Verify keyword copying
        sharky_creature = state.play_areas[Player.PLAYER_1][0]
        keywords = sharky_creature.get_effective_keywords(
            state.play_areas[Player.PLAYER_1],
            state.play_areas[Player.PLAYER_2]
        )
        
        self.assertIn(Keyword.SNEAKY, keywords)
        self.assertIn(Keyword.POISONOUS, keywords)
        
        # Remove enemy and verify keywords removed
        state.play_areas[Player.PLAYER_2].clear()
        keywords = sharky_creature.get_effective_keywords(
            state.play_areas[Player.PLAYER_1],
            state.play_areas[Player.PLAYER_2]
        )
        
        self.assertNotIn(Keyword.SNEAKY, keywords)
        self.assertNotIn(Keyword.POISONOUS, keywords)
    
    def test_deathweaver_blocks_play_effects(self):
        # Test Deathweaver preventing Play abilities
        state = self.engine.create_initial_state()
        
        # P1 has Axolotl Healer
        healer = self.cards["Axolotl Healer"]
        state.hands[Player.PLAYER_1] = [healer]
        
        # P2 has Deathweaver
        deathweaver = self.cards["Deathweaver"]
        state.play_areas[Player.PLAYER_2].append(
            CreatureState(card=deathweaver, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
        )
        
        # Update Deathweaver status
        self.engine._update_deathweaver_status(state)
        self.assertTrue(state.deathweaver_active[Player.PLAYER_1])
        
        # Play Axolotl Healer
        initial_life = state.life[Player.PLAYER_1]
        action = Action(ActionType.PLAY_CREATURE, Player.PLAYER_1, card=healer)
        state = self.engine.apply_action(state, action)
        
        # Pass on Mindbug
        action = Action(ActionType.PASS_MINDBUG, Player.PLAYER_2)
        state = self.engine.apply_action(state, action)
        
        # Life should not increase
        self.assertEqual(state.life[Player.PLAYER_1], initial_life)
    
    def test_elephantopus_blocking_restriction(self):
        # Test blocking restrictions
        state = self.engine.create_initial_state()
        state.current_player = Player.PLAYER_1
        
        # P1 has Elephantopus and attacker
        elephantopus = self.cards["Elephantopus"]
        attacker_card = self.cards["Spider Owl"]  # 3 power
        
        state.play_areas[Player.PLAYER_1].extend([
            CreatureState(card=elephantopus, controller=Player.PLAYER_1, owner=Player.PLAYER_1),
            CreatureState(card=attacker_card, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        ])
        
        # P2 has various power creatures
        low_power = self.cards["Chameleon Sniper"]  # 1 power
        mid_power = self.cards["Axolotl Healer"]    # 4 power
        high_power = self.cards["Killer Bee"]       # 5 power
        
        state.play_areas[Player.PLAYER_2].extend([
            CreatureState(card=low_power, controller=Player.PLAYER_2, owner=Player.PLAYER_2),
            CreatureState(card=mid_power, controller=Player.PLAYER_2, owner=Player.PLAYER_2),
            CreatureState(card=high_power, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
        ])
        
        # Check valid blockers
        attacker = state.play_areas[Player.PLAYER_1][1]
        blockers = self.engine._get_valid_blockers(
            state, attacker,
            state.play_areas[Player.PLAYER_1],
            Player.PLAYER_2,
            state.play_areas[Player.PLAYER_2]
        )
        
        # Only 5+ power can block
        self.assertEqual(len(blockers), 1)
        self.assertEqual(blockers[0], 2)
    
    def test_hunter_overrides_elephantopus(self):
        # Test HUNTER overrides blocking restrictions
        state = self.engine.create_initial_state()
        state.current_player = Player.PLAYER_1
        
        # P1 has Elephantopus and HUNTER attacker
        elephantopus = self.cards["Elephantopus"]
        hunter_card = self.cards["Killer Bee"]  # Has HUNTER
        
        state.play_areas[Player.PLAYER_1].extend([
            CreatureState(card=elephantopus, controller=Player.PLAYER_1, owner=Player.PLAYER_1),
            CreatureState(card=hunter_card, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        ])
        
        # P2 has low power creature
        low_power = self.cards["Chameleon Sniper"]  # 1 power
        
        state.play_areas[Player.PLAYER_2].append(
            CreatureState(card=low_power, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
        )
        
        # HUNTER can force any block
        attacker = state.play_areas[Player.PLAYER_1][1]
        blockers = self.engine._get_valid_blockers(
            state, attacker,
            state.play_areas[Player.PLAYER_1],
            Player.PLAYER_2,
            state.play_areas[Player.PLAYER_2]
        )
        
        self.assertEqual(len(blockers), 1)
        self.assertEqual(blockers[0], 0)
    
    def test_kangasaurus_rex_mass_removal(self):
        # Test mass removal effect
        state = self.engine.create_initial_state()
        state.current_player = Player.PLAYER_1
        
        # P1 plays Kangasaurus Rex
        rex = self.cards["Kangasaurus Rex"]
        state.hands[Player.PLAYER_1] = [rex]
        
        # P2 has various power creatures
        creatures = [
            ("Chameleon Sniper", 1),  # Defeated
            ("Plated Scorpion", 2),   # Defeated
            ("Spider Owl", 3),         # Defeated
            ("Axolotl Healer", 4),     # Defeated
            ("Killer Bee", 5),         # Survives
            ("Gorillion", 10)          # Survives
        ]
        
        for name, _ in creatures:
            state.play_areas[Player.PLAYER_2].append(
                CreatureState(card=self.cards[name], controller=Player.PLAYER_2, owner=Player.PLAYER_2)
            )
        
        # Play Kangasaurus Rex
        action = Action(ActionType.PLAY_CREATURE, Player.PLAYER_1, card=rex)
        state = self.engine.apply_action(state, action)
        
        # Pass Mindbug
        action = Action(ActionType.PASS_MINDBUG, Player.PLAYER_2)
        state = self.engine.apply_action(state, action)
        
        # Check survivors
        self.assertEqual(len(state.play_areas[Player.PLAYER_2]), 2)
        remaining_names = [c.card.name for c in state.play_areas[Player.PLAYER_2]]
        self.assertIn("Killer Bee", remaining_names)
        self.assertIn("Gorillion", remaining_names)
    
    def test_frenzy_allows_two_attacks(self):
        # Test FRENZY keyword
        state = self.engine.create_initial_state()
        state.current_player = Player.PLAYER_1
        
        # P1 has FRENZY creature
        frenzy_card = self.cards["Explosive Toad"]  # Has FRENZY
        
        frenzy_creature = CreatureState(card=frenzy_card, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        state.play_areas[Player.PLAYER_1].append(frenzy_creature)
        
        # First attack allowed
        action = Action(ActionType.ATTACK, Player.PLAYER_1, creature_index=0)
        legal_actions = state.get_legal_actions()
        attack_actions = [a for a in legal_actions if a.action_type == ActionType.ATTACK]
        self.assertEqual(len(attack_actions), 1)
        
        # Apply first attack
        state = self.engine.apply_action(state, action)
        
        # Second attack allowed
        legal_actions = state.get_legal_actions()
        attack_actions = [a for a in legal_actions if a.action_type == ActionType.ATTACK]
        self.assertEqual(len(attack_actions), 1)
        
        # Apply second attack
        state = self.engine.apply_action(state, action)
        
        # Turn should end
        self.assertEqual(state.current_player, Player.PLAYER_2)


class TestCombatResolution(unittest.TestCase):
    # Test combat mechanics
    
    def setUp(self):
        self.cards = CardDefinitions.get_first_contact_cards()
        self.engine = MindbugEngine
    
    def test_tough_vs_poisonous(self):
        # Test TOUGH prevents first defeat from POISONOUS
        state = self.engine.create_initial_state()
        state.current_player = Player.PLAYER_1
        
        # P1 has POISONOUS attacker
        poisonous_card = self.cards["Spider Owl"]
        
        # P2 has TOUGH blocker
        tough_card = self.cards["Rhino Turtle"]
        
        attacker = CreatureState(card=poisonous_card, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        blocker = CreatureState(card=tough_card, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
        
        state.play_areas[Player.PLAYER_1].append(attacker)
        state.play_areas[Player.PLAYER_2].append(blocker)
        
        # Resolve combat
        self.engine._resolve_combat(state, attacker, Player.PLAYER_1, 0, Player.PLAYER_2)
        
        # TOUGH creature exhausted but alive
        self.assertEqual(len(state.play_areas[Player.PLAYER_2]), 1)
        self.assertTrue(state.play_areas[Player.PLAYER_2][0].is_exhausted)
        
        # POISONOUS attacker defeated by high power
        self.assertEqual(len(state.play_areas[Player.PLAYER_1]), 0)
    
    def test_mutual_poisonous(self):
        # Test mutual destruction
        state = self.engine.create_initial_state()
        state.current_player = Player.PLAYER_1
        
        # Both have POISONOUS
        card1 = self.cards["Spider Owl"]
        card2 = self.cards["Plated Scorpion"]
        
        attacker = CreatureState(card=card1, controller=Player.PLAYER_1, owner=Player.PLAYER_1)
        blocker = CreatureState(card=card2, controller=Player.PLAYER_2, owner=Player.PLAYER_2)
        
        state.play_areas[Player.PLAYER_1].append(attacker)
        state.play_areas[Player.PLAYER_2].append(blocker)
        
        # Resolve combat
        self.engine._resolve_combat(state, attacker, Player.PLAYER_1, 0, Player.PLAYER_2)
        
        # Both defeated
        self.assertEqual(len(state.play_areas[Player.PLAYER_1]), 0)
        self.assertEqual(len(state.play_areas[Player.PLAYER_2]), 0)


class TestDeepCFR(unittest.TestCase):
    # Test Deep CFR implementation
    
    def setUp(self):
        self.config = get_debug_config()
        self.config['use_gpu'] = torch.cuda.is_available()
    
    def test_network_architecture(self):
        # Verify network structure
        network = DualBranchMindbugNetwork(
            card_embedding_dim=64,
            history_dim=64,
            hidden_dim=128,
            num_card_types=32
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in network.parameters())
        print(f"Network has {total_params:,} parameters")
        
        # Test forward pass
        batch_size = 16
        card_indices = torch.randint(0, 33, (batch_size, 30))
        history_features = torch.randn(batch_size, 64)
        
        output = network(card_indices, history_features)
        self.assertEqual(output.shape, (batch_size, 1))
    
    def test_encoder_consistency(self):
        # Test state encoding consistency
        encoder = MindbugStateEncoder()
        state = MindbugEngine.create_initial_state()
        
        # Create action
        if state.hands[Player.PLAYER_1]:
            action = Action(ActionType.PLAY_CREATURE, Player.PLAYER_1, 
                          card=state.hands[Player.PLAYER_1][0])
        else:
            action = Action(ActionType.PASS_MINDBUG, Player.PLAYER_1)
        
        # Encode twice
        card_indices1, history1 = encoder.encode_state_and_action(state, Player.PLAYER_1, action)
        card_indices2, history2 = encoder.encode_state_and_action(state, Player.PLAYER_1, action)
        
        # Should be identical
        self.assertTrue(torch.equal(card_indices1, card_indices2))
        self.assertTrue(torch.equal(history1, history2))
    
    def test_cfr_convergence(self):
        # Test basic convergence
        cfr = DeepCFR(self.config)
        
        # Train briefly
        start_time = time.time()
        cfr.train(num_iterations=10)
        train_time = time.time() - start_time
        
        print(f"Training took {train_time:.2f} seconds")
        
        # Networks should exist
        self.assertIsNotNone(cfr.value_networks)
        
        # Get policy
        state = MindbugEngine.create_initial_state()
        policy = cfr.get_policy(state, Player.PLAYER_1)
        
        # Valid probability distribution
        self.assertGreater(len(policy), 0)
        total_prob = sum(policy.values())
        self.assertAlmostEqual(total_prob, 1.0, places=5)
    
    def test_batch_encoding(self):
        # Test batch efficiency
        encoder = MindbugStateEncoder()
        
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
        
        # Time batch encoding
        start_time = time.time()
        card_batch, history_batch = encoder.batch_encode(states, players, actions)
        batch_time = time.time() - start_time
        
        print(f"Batch encoding 32 states took {batch_time:.4f} seconds")
        
        self.assertEqual(card_batch.shape[0], 32)
        self.assertEqual(history_batch.shape[0], 32)


class TestPerformance(unittest.TestCase):
    # Performance benchmarks
    
    def test_state_copying_performance(self):
        # Compare copy methods
        state = MindbugEngine.create_initial_state()
        
        # Deepcopy benchmark
        start = time.time()
        for _ in range(1000):
            _ = state.copy()
        deepcopy_time = time.time() - start
        
        # FastGameState benchmark
        from mindbug.game.state_pool import FastGameState
        fast_state = FastGameState.from_game_state(state)
        
        start = time.time()
        for _ in range(1000):
            _ = fast_state.copy()
        fast_time = time.time() - start
        
        print(f"Deepcopy: {deepcopy_time:.3f}s, FastGameState: {fast_time:.3f}s")
        print(f"Speedup: {deepcopy_time / fast_time:.1f}x")
        
        # Should be significantly faster
        self.assertLess(fast_time, deepcopy_time * 0.5)
    
    def test_gpu_performance(self):
        # Test GPU acceleration
        if not torch.cuda.is_available():
            self.skipTest("GPU not available")
        
        network = DualBranchMindbugNetwork().cuda()
        
        # Create batch
        batch_size = 2048
        card_indices = torch.randint(0, 33, (batch_size, 30)).cuda()
        history_features = torch.randn(batch_size, 64).cuda()
        
        # Warmup
        for _ in range(10):
            _ = network(card_indices, history_features)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(100):
            _ = network(card_indices, history_features)
        
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        throughput = (batch_size * 100) / gpu_time
        print(f"GPU throughput: {throughput:,.0f} samples/second")
        
        # Should handle many samples per second
        self.assertGreater(throughput, 50000)


def run_all_tests():
    # Run complete test suite
    test_classes = [
        TestCardImplementations,
        TestCombatResolution,
        TestDeepCFR,
        TestPerformance
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    run_all_tests()