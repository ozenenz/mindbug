#!/usr/bin/env python3
"""Play Mindbug interactively."""
import argparse
import random
from typing import Dict, List, Optional

from mindbug import DeepCFR, GameState, MindbugEngine, Player
from mindbug.core import Action, ActionType, Keyword
from mindbug.utils import get_quick_config


def print_game_state(state: GameState, show_hand: bool = True) -> None:
    """Display the current game state."""
    print("\n" + "=" * 70)
    print(f"Current Player: {state.current_player.name}")
    print(f"Life - P1: {state.life[Player.PLAYER_1]} | P2: {state.life[Player.PLAYER_2]}")
    print(
        f"Mindbugs - P1: {state.mindbugs_available[Player.PLAYER_1]} | "
        f"P2: {state.mindbugs_available[Player.PLAYER_2]}"
    )

    # Show play areas
    for player in [Player.PLAYER_1, Player.PLAYER_2]:
        print(f"\n{player.name} Creatures:")
        if not state.play_areas[player]:
            print("  (none)")
        else:
            for i, creature in enumerate(state.play_areas[player]):
                # Get effective stats
                is_active = state.current_player == player
                power = creature.get_effective_power(is_active, state.play_areas[player])
                keywords = creature.get_effective_keywords(
                    state.play_areas[player], state.play_areas[player.other()]
                )

                # Format output
                keyword_str = ", ".join(k.value for k in keywords) if keywords else "None"
                exhausted = " [EXHAUSTED]" if creature.is_exhausted else ""

                print(
                    f"  [{i}] {creature.card.name} - Power: {power}, "
                    f"Keywords: {keyword_str}{exhausted}"
                )

                if creature.card.ability_text:
                    print(f"      Ability: {creature.card.ability_text}")

    # Show hand for human player
    if show_hand and state.current_player == Player.PLAYER_1:
        print(f"\nYour hand ({len(state.hands[state.current_player])} cards):")
        if not state.hands[state.current_player]:
            print("  (empty)")
        else:
            for i, card in enumerate(state.hands[state.current_player]):
                keyword_str = ", ".join(k.value for k in card.keywords) if card.keywords else "None"
                print(f"  [{i}] {card.name} (Power: {card.power}, Keywords: {keyword_str})")
                if card.ability_text:
                    print(f"      {card.ability_text}")

    print("=" * 70)


def get_human_action(state: GameState) -> Action:
    """Get action from human player."""
    actions = state.get_legal_actions()

    # Mindbug decision
    if state.mindbug_decision_pending:
        player, card = state.pending_creature_play
        print(f"\n{player.name} is playing {card.name}!")
        print("\nDo you want to use a Mindbug to steal it?")
        print("[0] Pass")
        print("[1] Use Mindbug (steal the creature)")

        while True:
            try:
                choice = int(input("\nChoice: "))
                if choice == 0:
                    return next(a for a in actions if a.action_type == ActionType.PASS_MINDBUG)
                elif choice == 1 and state.mindbugs_available[state.current_player] > 0:
                    return next(a for a in actions if a.action_type == ActionType.USE_MINDBUG)
                else:
                    print("Invalid choice or no Mindbugs available!")
            except (ValueError, StopIteration):
                print("Invalid input!")

    # Hunter choice
    if state.hunter_choice_pending:
        print("\nChoose which creature must block (HUNTER ability):")
        opponent = state.current_player.other()
        for i, idx in enumerate(state.valid_blockers):
            creature = state.play_areas[opponent][idx]
            power = creature.get_effective_power(
                state.current_player == opponent, state.play_areas[opponent]
            )
            print(f"[{i}] Force {creature.card.name} (Power: {power}) to block")

        while True:
            try:
                choice = int(input("\nChoice: "))
                if 0 <= choice < len(actions):
                    return actions[choice]
                else:
                    print("Invalid choice!")
            except ValueError:
                print("Invalid input!")

    # Normal turn
    print("\nAvailable actions:")

    # Group actions by type
    play_actions = []
    attack_actions = []

    for action in actions:
        if action.action_type == ActionType.PLAY_CREATURE:
            play_actions.append(action)
        elif action.action_type == ActionType.ATTACK:
            attack_actions.append(action)

    # Display options
    if play_actions:
        print("\nPlay a creature from hand:")
        for i, action in enumerate(play_actions):
            card = action.card
            keyword_str = ", ".join(k.value for k in card.keywords) if card.keywords else "None"
            print(f"  [P{i}] {card.name} (Power: {card.power}, Keywords: {keyword_str})")

    if attack_actions:
        print("\nAttack with a creature:")
        for i, action in enumerate(attack_actions):
            creature = state.play_areas[state.current_player][action.creature_index]
            power = creature.get_effective_power(True, state.play_areas[state.current_player])
            print(f"  [A{i}] Attack with {creature.card.name} (Power: {power})")

    # Get choice
    while True:
        choice = input("\nEnter choice (e.g., P0 to play, A0 to attack): ").strip().upper()

        try:
            if choice.startswith("P") and play_actions:
                idx = int(choice[1:])
                if 0 <= idx < len(play_actions):
                    return play_actions[idx]
            elif choice.startswith("A") and attack_actions:
                idx = int(choice[1:])
                if 0 <= idx < len(attack_actions):
                    return attack_actions[idx]
        except (ValueError, IndexError):
            pass

        print("Invalid choice! Use P# for play or A# for attack.")


def get_ai_action(state: GameState, agent: Optional[DeepCFR]) -> Action:
    """Get action from AI agent."""
    if agent:
        # Use trained agent
        policy = agent.get_policy(state, state.current_player)

        # Sample from policy
        actions = list(policy.keys())
        probs = list(policy.values())

        # Normalize
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            probs = [1.0 / len(actions)] * len(actions)

        return random.choices(actions, weights=probs)[0]
    else:
        # Random AI
        actions = state.get_legal_actions()
        return random.choice(actions)


def format_action(action: Action, state: GameState) -> str:
    """Format action for display."""
    if action.action_type == ActionType.PLAY_CREATURE:
        return f"plays {action.card.name}"
    elif action.action_type == ActionType.ATTACK:
        creature = state.play_areas[action.player][action.creature_index]
        return f"attacks with {creature.card.name}"
    elif action.action_type == ActionType.USE_MINDBUG:
        return "uses Mindbug!"
    elif action.action_type == ActionType.PASS_MINDBUG:
        return "passes on Mindbug"
    elif action.action_type == ActionType.CHOOSE_BLOCKER:
        opponent = action.player.other()
        blocker = state.play_areas[opponent][action.target_index]
        return f"forces {blocker.card.name} to block"
    else:
        return str(action)


def play_game(
    agent: Optional[DeepCFR] = None,
    human_player: Player = Player.PLAYER_1,
    show_ai_state: bool = False,
) -> None:
    """Play an interactive game."""
    print("\nStarting new game...")
    print(f"You are playing as {human_player.name}")

    state = MindbugEngine.create_initial_state()

    while not state.is_terminal():
        # Show state
        is_human_turn = state.current_player == human_player
        print_game_state(state, show_hand=is_human_turn or show_ai_state)

        # Get action
        if is_human_turn:
            action = get_human_action(state)
        else:
            print(f"\n{state.current_player.name} is thinking...")
            action = get_ai_action(state, agent)
            print(f"{state.current_player.name} {format_action(action, state)}")

            if not state.mindbug_decision_pending and not state.hunter_choice_pending:
                input("\nPress Enter to continue...")

        # Apply action
        state = MindbugEngine.apply_action(state, action)

    # Game over
    print_game_state(state, show_hand=False)
    winner = state.get_winner()

    print("\n" + "=" * 70)
    if winner is None:
        print("Game ended in a DRAW!")
    elif winner == human_player:
        print("CONGRATULATIONS! You won!")
    else:
        print("You lost. Better luck next time!")
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Play Mindbug against AI", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--player", choices=["1", "2"], default="1", help="Which player to play as (default: 1)"
    )
    parser.add_argument(
        "--show-ai-hand", action="store_true", help="Show AI's hand (for debugging)"
    )

    args = parser.parse_args()

    # Load agent if checkpoint provided
    agent = None
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        config = get_quick_config()
        config["use_gpu"] = False  # Use CPU for interactive play

        agent = DeepCFR(config)
        try:
            agent.load_checkpoint(args.checkpoint)
            print("Loaded trained agent!")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Using random AI instead.")
            agent = None
    else:
        print("No checkpoint provided. Playing against random AI.")

    # Determine human player
    human_player = Player.PLAYER_1 if args.player == "1" else Player.PLAYER_2

    # Play game
    print("\nWelcome to Mindbug!")
    print("=" * 70)

    while True:
        play_game(agent, human_player, args.show_ai_hand)

        again = input("\nPlay again? (y/n): ").strip().lower()
        if again != "y":
            break

    print("\nThanks for playing!")


if __name__ == "__main__":
    main()
