import copy
import numpy as np
from typing import Tuple
from classes.logic import BLACK_PLAYER, WHITE_PLAYER, get_possible_moves, is_game_over, is_node_free
from classes.ui import UI
import classes.logic as logic


class PlayerStrat:
    def __init__(self, _board_state, player):
        self.root_state = _board_state
        self.player = player

    def start(self) -> Tuple[int, int]:
        raise NotImplementedError


class Node:
    def __init__(self, state, move=None, wins=0, visits=0, children=None):
        self.state = state
        self.move = move
        self.wins = wins
        self.visits = visits
        self.children = children or []
        self.parent = None
        self.untried_moves = get_possible_moves(state)

    def add_child(self, child):
        child.parent = self
        self.children.append(child)


class Random(PlayerStrat):
    def start(self) -> Tuple[int, int]:
        legal_moves = get_possible_moves(self.root_state)
        return legal_moves[np.random.choice(len(legal_moves))]


class MiniMax(PlayerStrat):
    def start(self) -> Tuple[int, int]:
        _, move = self.minimax(self.root_state, depth=3, maximizing_player=True, alpha=float('-inf'), beta=float('inf'), current_player=self.player)
        return move

    def minimax(self, state: np.ndarray, depth: int, maximizing_player: bool, alpha: float, beta: float, current_player: int) -> Tuple[int, Tuple[int, int]]:
        if depth == 0 or logic.is_game_over(current_player, state):
            return self.evaluate(state, current_player), None

        legal_moves = logic.get_possible_moves(state)
        best_move = None

        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                if not logic.is_node_free(move, state):
                    continue

                new_state = copy.deepcopy(state)
                new_state[move] = current_player
                eval, _ = self.minimax(new_state, depth - 1, False, alpha, beta, 3 - current_player)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in legal_moves:
                if not logic.is_node_free(move, state):
                    continue

                new_state = copy.deepcopy(state)
                new_state[move] = 3 - current_player
                eval, _ = self.minimax(new_state, depth - 1, True, alpha, beta, 3 - current_player)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def evaluate(self, state: np.ndarray, current_player: int) -> int:
        winner = logic.is_game_over(current_player, state)
        opponent_winner = logic.is_game_over(3 - current_player, state)

        if winner:
            return 1
        elif opponent_winner:
            return -1
        else:
            return 0



str2strat: dict[str, PlayerStrat] = {
    "human": None,
    "random": Random,
    "minimax": MiniMax,
}
