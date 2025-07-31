import random
from typing import Dict, List, Optional

from mindbug.game import (
    MindbugEngine, GameState, Player, Action, ActionType,
    Card, CardDefinitions, Keyword
)
from mindbug.algorithms import DeepCFR


def print_game_state(state: GameState):
    # Display current game state
    print("\n" + "=" * 60)
    print(f"Current Player: {state.current_player.name}")
    print(f"Life - P1: {state.life[Player.PLAYER_1]} | P2: {state.life[Player.PLAYER_2]}")
    print(f"Mindbugs - P1: {state.mindbugs_available[Player.PLAYER_1]} | P2: {state.mindbugs_available[Player.PLAYER_2]}")
    
    # Show play areas
    for player in [Player.PLAYER_1, Player.PLAYER_2]:
        print(f"\n{player.name} Creatures:")
        for i, creature in enumerate(state.play_areas[player]):
            keywords = creature.get_effective_keywords(
                state.play_areas[player],
                state.play_areas[player.other()]
            )
            power = creature.get_effective_power(
                state.current_player == player,
                state.play_areas[player]
            )
            keyword_str = ", ".join(k.value for k in keywords) if keywords else "None"
            exhausted = " (Exhausted)" if creature.is_exhausted else ""
            print(f"  [{i}] {creature.card.name} - Power: {power}, Keywords: {keyword_str}{exhausted}")
    
    # Show hand for current player
    if state.current_player == Player.PLAYER_1:  # Human player
        print(f"\nYour hand:")
        for i, card in enumerate(state.hands[state.current_player]):
            keyword_str = ", ".join(k.value for k in card.keywords) if card.keywords else "None"
            ability = f" - {card.ability_text}" if card.ability_text else ""
            print(f"  [{i}] {card.name} (Power: {card.power}, Keywords: {keyword_str}){ability}")
    
    print("=" * 60)


def get_human_action(state: GameState) -> Action:
    # Get action from human player
    actions = state.get_legal_actions()
    
    # Special phases
    if state.mindbug_decision_pending:
        print("\nOpponent played a creature! Use Mindbug?")
        print("[0] Pass")
        print("[1] Use Mindbug")
        
        while True:
            try:
                choice = int(input("Choice: "))
                if choice == 0:
                    return actions[1]  # Pass
                elif choice == 1 and len(actions) > 1:
                    return actions[0]  # Use Mindbug
            except (ValueError, IndexError):
                pass
            print("Invalid choice!")
    
    if state.hunter_choice_pending:
        print("\nChoose blocker for HUNTER:")
        for i, idx in enumerate(state.valid_blockers):
            creature = state.play_areas[state.current_player.other()][idx]
            print(f"[{i}] {creature.card.name} (Power: {creature.card.power})")
        
        while True:
            try:
                choice = int(input("Choice: "))
                return actions[choice]
            except (ValueError, IndexError):
                pass
            print("Invalid choice!")
    
    # Normal turn
    print("\nAvailable actions:")
    
    # Group actions by type
    play_actions = []
    attack_actions = []
    
    for i, action in enumerate(actions):
        if action.action_type == ActionType.PLAY_CREATURE:
            play_actions.append((i, action))
        elif action.action_type == ActionType.ATTACK:
            attack_actions.append((i, action))
    
    # Display play options
    if play_actions:
        print("\nPlay creature:")
        for i, (idx, action) in enumerate(play_actions):
            card = action.card
            keyword_str = ", ".join(k.value for k in card.keywords) if card.keywords else "None"
            ability = f" - {card.ability_text}" if card.ability_text else ""
            print(f"  [P{i}] {card.name} (Power: {card.power}, Keywords: {keyword_str}){ability}")
    
    # Display attack options
    if attack_actions:
        print("\nAttack with creature:")
        for i, (idx, action) in enumerate(attack_actions):
            creature = state.play_areas[state.current_player][action.creature_index]
            print(f"  [A{i}] {creature.card.name}")
    
    # Get choice
    while True:
        choice = input("\nEnter choice (e.g., P0 to play first creature, A0 to attack): ").strip().upper()
        
        try:
            if choice.startswith('P') and play_actions:
                idx = int(choice[1:])
                return play_actions[idx][1]
            elif choice.startswith('A') and attack_actions:
                idx = int(choice[1:])
                return attack_actions[idx][1]
        except (ValueError, IndexError):
            pass
        
        print("Invalid choice!")


def get_ai_action(state: GameState, agent: Optional[DeepCFR]) -> Action:
    # Get action from AI agent
    if agent:
        # Use trained agent
        policy = agent.get_policy(state, state.current_player)
        actions = list(policy.keys())
        probs = list(policy.values())
        return random.choices(actions, weights=probs)[0]
    else:
        # Random AI
        actions = state.get_legal_actions()
        return random.choice(actions)


def play_interactive_game(agent: Optional[DeepCFR] = None, human_player: Player = Player.PLAYER_1):
    # Play interactive game against AI
    print("Starting Mindbug game!")
    print(f"You are playing as {human_player.name}")
    
    state = MindbugEngine.create_initial_state()
    
    while not state.is_terminal():
        print_game_state(state)
        
        if state.current_player == human_player:
            action = get_human_action(state)
        else:
            action = get_ai_action(state, agent)
            print(f"\nAI plays: {action}")
        
        state = MindbugEngine.apply_action(state, action)
    
    # Game over
    print_game_state(state)
    winner = state.get_winner()
    
    if winner is None:
        print("\nGame ended in a draw!")
    elif winner == human_player:
        print("\nCongratulations! You won!")
    else:
        print("\nAI wins! Better luck next time!")


def main():
    # Main entry point
    print("Welcome to Mindbug!")
    print("\nOptions:")
    print("[1] Play against random AI")
    print("[2] Play against trained AI (requires checkpoint)")
    print("[3] Watch AI vs AI")
    
    while True:
        try:
            choice = int(input("\nChoice: "))
            if choice in [1, 2, 3]:
                break
        except ValueError:
            pass
        print("Invalid choice!")
    
    agent = None
    if choice == 2:
        # Load trained agent
        checkpoint_path = input("Enter checkpoint path: ").strip()
        try:
            agent = DeepCFR({"use_gpu": False})  # Simple config
            agent.load_checkpoint(checkpoint_path)
            print("Loaded trained agent!")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Using random AI instead.")
    
    if choice == 3:
        # AI vs AI
        print("\nWatching AI vs AI game...")
        state = MindbugEngine.create_initial_state()
        
        while not state.is_terminal():
            print_game_state(state)
            action = get_ai_action(state, agent)
            print(f"\n{state.current_player.name} plays: {action}")
            state = MindbugEngine.apply_action(state, action)
            input("Press Enter to continue...")
        
        print_game_state(state)
        winner = state.get_winner()
        if winner:
            print(f"\n{winner.name} wins!")
        else:
            print("\nDraw!")
    else:
        # Human vs AI
        play_interactive_game(agent)


if __name__ == "__main__":
    main()