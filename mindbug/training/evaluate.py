from typing import Dict, Tuple

import numpy as np

from ..algorithms.deep_cfr import DeepCFR
from ..game.constants import Player
from ..game.engine import MindbugEngine
from ..game.state import GameState


class Evaluator:
    @staticmethod
    def play_game(agent1: DeepCFR, agent2: DeepCFR) -> Player:
        state = MindbugEngine.create_initial_state()
        while not state.is_terminal():
            current_player = state.current_player
            if current_player == Player.PLAYER_1:
                policy = agent1.get_policy(state, current_player)
            else:
                policy = agent2.get_policy(state, current_player)
            actions = list(policy.keys())
            probs = list(policy.values())
            action = np.random.choice(actions, p=probs)
            state = MindbugEngine.apply_action(state, action)
        return state.get_winner()

    @staticmethod
    def evaluate_agents(
        agent1: DeepCFR, agent2: DeepCFR, num_games: int = 1000
    ) -> Dict[str, float]:
        wins = {Player.PLAYER_1: 0, Player.PLAYER_2: 0, None: 0}
        for i in range(num_games):
            if i % 2 == 0:
                winner = Evaluator.play_game(agent1, agent2)
            else:
                winner = Evaluator.play_game(agent2, agent1)
                if winner == Player.PLAYER_1:
                    winner = Player.PLAYER_2
                elif winner == Player.PLAYER_2:
                    winner = Player.PLAYER_1
            wins[winner] += 1
        return {
            "agent1_win_rate": wins[Player.PLAYER_1] / num_games,
            "agent2_win_rate": wins[Player.PLAYER_2] / num_games,
            "draw_rate": wins[None] / num_games,
        }

    @staticmethod
    def compute_exploitability(agent: DeepCFR, num_iterations: int = 1000) -> float:
        total_utility = 0.0
        for _ in range(num_iterations):
            state = MindbugEngine.create_initial_state()
            while not state.is_terminal():
                current_player = state.current_player
                legal_actions = state.get_legal_actions()
                if current_player == Player.PLAYER_1:
                    policy = agent.get_policy(state, current_player)
                    actions = list(policy.keys())
                    probs = list(policy.values())
                    action = np.random.choice(actions, p=probs)
                else:
                    action = np.random.choice(legal_actions)
                state = MindbugEngine.apply_action(state, action)
            winner = state.get_winner()
            if winner == Player.PLAYER_1:
                total_utility += 1.0
            elif winner == Player.PLAYER_2:
                total_utility -= 1.0
        return -total_utility / num_iterations
