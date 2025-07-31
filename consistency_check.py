# Verify all imports and interfaces work correctly
import sys
from typing import List, Tuple

def check_imports() -> Tuple[List[str], List[str]]:
    # Check all imports work correctly
    successful = []
    failed = []
    
    imports_to_check = [
        # Core imports
        ("mindbug", ["MindbugEngine", "GameState", "Player", "DeepCFR"]),
        
        # Game module
        ("mindbug.game", ["MindbugEngine", "GameState", "Action", "Card", 
                          "CardDefinitions", "Player", "ActionType", "Keyword"]),
        ("mindbug.game.engine", ["MindbugEngine"]),
        ("mindbug.game.state", ["GameState", "CreatureState"]),
        ("mindbug.game.actions", ["Action"]),
        ("mindbug.game.cards", ["Card", "CardDefinitions"]),
        ("mindbug.game.constants", ["Player", "ActionType", "Keyword", "TriggerType"]),
        
        # Algorithms module
        ("mindbug.algorithms", ["DeepCFR", "DualBranchMindbugNetwork", "MindbugStateEncoder"]),
        ("mindbug.algorithms.deep_cfr", ["DeepCFR"]),
        ("mindbug.algorithms.networks", ["DualBranchMindbugNetwork", "MindbugStateEncoder"]),
        
        # Training module
        ("mindbug.training", ["train", "Evaluator"]),
        ("mindbug.training.train", ["train"]),
        ("mindbug.training.evaluate", ["Evaluator"]),
        
        # Utils module
        ("mindbug.utils.config", ["get_quick_test_config", "get_performance_config",
                                  "get_distributed_config", "get_debug_config"]),
        
        # Test framework
        ("mindbug.test_framework", ["TestCardImplementations", "TestCombatResolution",
                                    "TestDeepCFR", "TestPerformance", "run_all_tests"]),
    ]
    
    for module_name, expected_attrs in imports_to_check:
        try:
            module = __import__(module_name, fromlist=expected_attrs)
            for attr in expected_attrs:
                if not hasattr(module, attr):
                    failed.append(f"{module_name}.{attr} - not found")
                else:
                    successful.append(f"{module_name}.{attr}")
        except ImportError as e:
            failed.append(f"{module_name} - {str(e)}")
    
    return successful, failed


def check_game_consistency():
    # Verify game components work together
    from mindbug.game import MindbugEngine, GameState, Player, Action, ActionType
    from mindbug.game.cards import CardDefinitions
    
    print("\nChecking game consistency...")
    
    # Create initial state
    state = MindbugEngine.create_initial_state()
    assert isinstance(state, GameState)
    assert state.life[Player.PLAYER_1] == 3
    assert state.life[Player.PLAYER_2] == 3
    print("✓ Initial state creation")
    
    # Check legal actions
    actions = state.get_legal_actions()
    assert len(actions) > 0
    assert all(isinstance(a, Action) for a in actions)
    print("✓ Legal actions generation")
    
    # Verify all cards exist
    cards = CardDefinitions.get_first_contact_cards()
    assert len(cards) == 32
    deck = CardDefinitions.get_first_contact_deck()
    assert len(deck) == 48
    print("✓ Card definitions")
    
    # Test action application
    if actions:
        action = actions[0]
        new_state = MindbugEngine.apply_action(state, action)
        assert isinstance(new_state, GameState)
        assert new_state != state  # Should be different object
        print("✓ Action application")


def check_deep_cfr_consistency():
    # Verify Deep CFR components work together
    from mindbug.algorithms import DeepCFR, MindbugStateEncoder
    from mindbug.utils.config import get_debug_config
    from mindbug.game import MindbugEngine, Player, Action, ActionType
    
    print("\nChecking Deep CFR consistency...")
    
    # Create Deep CFR instance
    config = get_debug_config()
    cfr = DeepCFR(config)
    print("✓ Deep CFR initialization")
    
    # Test encoder
    encoder = MindbugStateEncoder()
    state = MindbugEngine.create_initial_state()
    action = Action(ActionType.PASS_MINDBUG, Player.PLAYER_1)
    
    card_indices, history_features = encoder.encode_state_and_action(
        state, Player.PLAYER_1, action
    )
    assert card_indices.shape[0] == 30
    assert history_features.shape[0] == 64
    print("✓ State encoding")
    
    # Test policy generation
    policy = cfr.get_policy(state, Player.PLAYER_1)
    assert isinstance(policy, dict)
    assert sum(policy.values()) > 0.99  # Should sum to ~1
    print("✓ Policy generation")


def check_training_consistency():
    # Verify training components work together
    from mindbug.training import Evaluator
    from mindbug.algorithms import DeepCFR
    from mindbug.utils.config import get_debug_config
    
    print("\nChecking training consistency...")
    
    # Create agents
    config = get_debug_config()
    agent1 = DeepCFR(config)
    agent2 = DeepCFR(config)
    
    # Test evaluation
    results = Evaluator.evaluate_agents(agent1, agent2, num_games=10)
    assert "agent1_win_rate" in results
    assert "agent2_win_rate" in results
    assert "draw_rate" in results
    assert abs(sum(results.values()) - 1.0) < 0.001
    print("✓ Agent evaluation")
    
    # Test exploitability
    exploitability = Evaluator.compute_exploitability(agent1, num_iterations=10)
    assert isinstance(exploitability, float)
    print("✓ Exploitability computation")


def main():
    print("Running Mindbug Deep CFR Consistency Checks...")
    print("=" * 60)
    
    # Check imports
    successful, failed = check_imports()
    
    if failed:
        print(f"\n❌ Import failures ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
    else:
        print(f"\n✅ All {len(successful)} imports successful!")
    
    # Check component consistency
    try:
        check_game_consistency()
        check_deep_cfr_consistency()
        check_training_consistency()
        print("\n✅ All consistency checks passed!")
    except Exception as e:
        print(f"\n❌ Consistency check failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 60)
    print("Codebase is consistent and ready for use!")
    return 0


if __name__ == "__main__":
    sys.exit(main())