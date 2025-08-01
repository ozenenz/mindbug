"""Agent evaluation utilities."""

from typing import Dict, Optional

import numpy as np

from ..algorithms import DeepCFR
from ..core import GameState, MindbugEngine, Player


class Evaluator:
    """Methods for evaluating trained agents."""

    @staticmethod
    def play_game(
        agent1: DeepCFR, agent2: DeepCFR, starting_player: Optional[Player] = None
    ) -> Optional[Player]:
        """Play one game between two agents."""
        state = MindbugEngine.create_initial_state(starting_player=starting_player)

        agents = {Player.PLAYER_1: agent1, Player.PLAYER_2: agent2}

        while not state.is_terminal():
            current_player = state.current_player
            agent = agents[current_player]

            # Get policy from agent
            policy = agent.get_policy(state, current_player)

            # Sample action
            actions = list(policy.keys())
            probs = list(policy.values())

            # Ensure valid probability distribution
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            else:
                probs = [1.0 / len(actions)] * len(actions)

            action = np.random.choice(actions, p=probs)

            # Apply action
            state = MindbugEngine.apply_action(state, action)

        return state.get_winner()

    @staticmethod
    def evaluate_agents(
        agent1: DeepCFR, agent2: DeepCFR, num_games: int = 1000
    ) -> Dict[str, float]:
        """Evaluate win rates between two agents."""
        wins = {Player.PLAYER_1: 0, Player.PLAYER_2: 0, None: 0}  # Draws

        for i in range(num_games):
            # Alternate starting positions for fairness
            starting_player = Player.PLAYER_1 if i % 2 == 0 else Player.PLAYER_2

            if i % 2 == 0:
                # Agent1 is Player 1
                winner = Evaluator.play_game(agent1, agent2, starting_player)
                if winner == Player.PLAYER_1:
                    wins[Player.PLAYER_1] += 1
                elif winner == Player.PLAYER_2:
                    wins[Player.PLAYER_2] += 1
                else:
                    wins[None] += 1
            else:
                # Agent2 is Player 1 (swap positions)
                winner = Evaluator.play_game(agent2, agent1, starting_player)
                if winner == Player.PLAYER_1:
                    wins[Player.PLAYER_2] += 1  # Agent2 won
                elif winner == Player.PLAYER_2:
                    wins[Player.PLAYER_1] += 1  # Agent1 won
                else:
                    wins[None] += 1

        return {
            "agent1_win_rate": wins[Player.PLAYER_1] / num_games,
            "agent2_win_rate": wins[Player.PLAYER_2] / num_games,
            "draw_rate": wins[None] / num_games,
        }

    @staticmethod
    def compute_exploitability(agent: DeepCFR, num_iterations: int = 1000) -> float:
        """Estimate exploitability against random policy."""
        total_utility = 0.0

        for i in range(num_iterations):
            state = MindbugEngine.create_initial_state()

            # Track the agent's position
            agent_player = Player.PLAYER_1 if i % 2 == 0 else Player.PLAYER_2

            while not state.is_terminal():
                current_player = state.current_player
                legal_actions = state.get_legal_actions()

                if current_player == agent_player:
                    # Agent plays optimally
                    policy = agent.get_policy(state, current_player)
                    actions = list(policy.keys())
                    probs = list(policy.values())

                    # Normalize
                    total = sum(probs)
                    if total > 0:
                        probs = [p / total for p in probs]
                    else:
                        probs = [1.0 / len(actions)] * len(actions)

                    action = np.random.choice(actions, p=probs)
                else:
                    # Random opponent
                    action = np.random.choice(legal_actions)

                state = MindbugEngine.apply_action(state, action)

            # Compute utility for agent
            winner = state.get_winner()
            if winner == agent_player:
                total_utility += 1.0
            elif winner == agent_player.other():
                total_utility -= 1.0
            # Draw = 0

        # Average utility
        avg_utility = total_utility / num_iterations

        # Exploitability is how much the agent loses against best response
        # For a Nash equilibrium, this should be 0
        # We estimate it as negative of performance against random
        return -avg_utility

    @staticmethod
    def self_play_evaluation(agent: DeepCFR, num_games: int = 100) -> Dict[str, float]:
        """Evaluate agent in self-play."""
        return Evaluator.evaluate_agents(agent, agent, num_games)
