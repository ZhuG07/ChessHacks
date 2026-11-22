"""
ChessHacks tournament entrypoint.

The ChessHacks platform (and the local devtools via serve.py) imports this
module and calls the function registered with @chess_manager.entrypoint
every time the bot has to move. All the real work lives in:

    src/encoding.py  board -> tensor, move -> policy index
    src/model.py     network definitions + checkpoint loading
    src/search.py    alpha-beta search guided by the network
    src/engine.py    ties the three together

This file only adapts that engine to the tournament interface, and keeps
the names the self-play scripts import (PolicyOnlyEngine, board_to_planes,
move_to_index, POLICY_DIM).
"""

import random

import chess
from chess import Move

from .encoding import NUM_PLANES, POLICY_DIM, board_to_planes, move_to_index  # noqa: F401
from .engine import Engine, resolve_checkpoint
from .search import SearchConfig
from .utils import GameContext, chess_manager

MODEL_PATH = resolve_checkpoint()

# Per-move budget. Tournament games are played with a clock, so the engine
# uses a fixed slice of whatever time is left, within these bounds.
MIN_THINK_SECONDS = 0.1
MAX_THINK_SECONDS = 2.0
CLOCK_FRACTION = 1 / 30


class PolicyOnlyEngine:
    """
    Backwards-compatible wrapper used by selfPlay/. Exposes the old
    search_best_move(board, max_depth) -> (move, probs) interface.
    """

    def __init__(self, checkpoint=None, time_limit: float | None = 0.5):
        self._engine = Engine(checkpoint, config=SearchConfig(time_limit=time_limit))
        self.model = self._engine.model
        self.device = self._engine.device

    def search_best_move(self, board: chess.Board, max_depth: int):
        result = self._engine.analyse(board, max_depth=max_depth)
        return result.best_move, result.move_probabilities()

    def select_move_policy_only(self, board: chess.Board):
        result = self._engine.instinct(board)
        return result.best_move, result.policy

    def reset(self):
        self._engine.new_game()


try:
    ENGINE = Engine(MODEL_PATH)
    print(f"[ENGINE] Loaded {ENGINE.describe()}")
except Exception as e:  # keep the bot alive even if the weights are missing
    ENGINE = None
    print(f"[ENGINE] Failed to load network ({e}); falling back to random moves.")


def _think_time(time_left_ms: int) -> float:
    if not time_left_ms or time_left_ms <= 0:
        return 0.5
    budget = (time_left_ms / 1000) * CLOCK_FRACTION
    return max(MIN_THINK_SECONDS, min(MAX_THINK_SECONDS, budget))


def _random_move(ctx: GameContext, legal_moves: list[Move]) -> Move:
    p = 1.0 / len(legal_moves)
    ctx.logProbabilities({mv: p for mv in legal_moves})
    return random.choice(legal_moves)


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    board = ctx.board
    legal_moves = list(board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    if ENGINE is None:
        return _random_move(ctx, legal_moves)

    try:
        result = ENGINE.analyse(board, time_limit=_think_time(ctx.timeLeft))
    except Exception as e:
        print(f"[ENGINE] Search failed ({e}); using the raw policy instead.")
        result = ENGINE.instinct(board)

    if result.best_move is None:
        return _random_move(ctx, legal_moves)

    probs = result.move_probabilities()
    probs = {mv: float(probs.get(mv, 0.0)) for mv in legal_moves}
    total = sum(probs.values()) or 1.0
    ctx.logProbabilities({mv: p / total for mv, p in probs.items()})

    print(f"[ENGINE] {board.san(result.best_move)} score={result.score:.0f} "
          f"depth={result.depth} nodes={result.nodes} {result.time_ms:.0f}ms")
    return result.best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    if ENGINE is not None:
        ENGINE.new_game()
