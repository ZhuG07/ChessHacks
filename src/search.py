"""
Search on top of the policy/value network.

The network gives two things for any position:
  * a value (centipawn-like score, side-to-move point of view)
  * a policy (a prior over moves, trained to predict Stockfish's choice)

The search is a classic negamax with alpha-beta pruning, wrapped in
iterative deepening, with a few changes that make sense when the static
evaluation is an expensive neural network rather than a cheap handwritten
function:

  * Batched leaf evaluation. At a frontier node (depth 1) the children are
    sent to the network in batches instead of one at a time. On a CPU a
    batch of 16 positions costs only a few times more than a single one.
  * Quiescence search. At the leaves, captures that do not obviously lose
    material are played out before the network is trusted, so the engine
    does not stop thinking in the middle of an exchange (the horizon
    effect).
  * Policy move ordering. Moves are tried in order of the network's prior,
    so alpha-beta gets cutoffs early. The transposition table move, when
    there is one, goes first.
  * Policy pruning. Away from the root, only the most likely moves (plus
    every capture, promotion and check, and every move when in check) are
    searched. This trades a little safety for a lot of extra depth.
  * MultiPV at the root, so the UI can show the top few lines with exact
    scores rather than just a single best move.
  * Anytime behaviour. Each completed iteration is a valid answer; when the
    clock runs out mid-iteration the last completed one is returned.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import chess
import chess.polyglot
import numpy as np
import torch

from .encoding import board_to_planes, move_to_index

MATE_SCORE = 100_000
MATE_THRESHOLD = MATE_SCORE - 1_000
INF = float("inf")

# Rough material values, used only to order and filter captures in the
# quiescence search. The evaluation itself always comes from the network.
PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20_000,
}
DELTA_MARGIN = 200


if hasattr(chess.Board, "_transposition_key"):
    def position_key(board: chess.Board):
        """Hashable key for a position (pieces, side to move, castling, ep)."""
        return board._transposition_key()
else:  # pragma: no cover - older python-chess
    position_key = chess.polyglot.zobrist_hash

EXACT, LOWER, UPPER = 0, 1, 2


class SearchTimeout(Exception):
    """Raised inside the tree when the time budget is spent."""


# ============================================================
# Results
# ============================================================

@dataclass
class Line:
    move: chess.Move
    score: float               # centipawns, side to move at the root
    pv: list[chess.Move]
    prior: float               # raw network policy probability
    exact: bool = True         # False if the score is only an upper bound


@dataclass
class SearchResult:
    best_move: chess.Move | None
    score: float
    depth: int
    nodes: int
    nn_evals: int
    time_ms: float
    pv: list[chess.Move] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)
    policy: dict[chess.Move, float] = field(default_factory=dict)
    static_eval: float = 0.0   # network value of the root before searching
    win_prob: float = 0.5      # from the result head, side to move

    @property
    def mate_in(self) -> int | None:
        return mate_distance(self.score)

    def move_probabilities(self, temperature: float = 120.0) -> dict[chess.Move, float]:
        """Softmax over root scores, for logging and visualisation."""
        if not self.lines:
            return dict(self.policy)
        scores = np.array([max(min(l.score, 5_000), -5_000) for l in self.lines], dtype=np.float64)
        scores = (scores - scores.max()) / temperature
        probs = np.exp(scores)
        probs /= probs.sum()
        return {l.move: float(p) for l, p in zip(self.lines, probs)}


def mate_distance(score: float) -> int | None:
    """Moves to mate (positive: side to move mates, negative: gets mated)."""
    if score >= MATE_THRESHOLD:
        return int((MATE_SCORE - score + 1) // 2)
    if score <= -MATE_THRESHOLD:
        return -int((MATE_SCORE + score + 1) // 2)
    return None


# ============================================================
# Network wrapper with a cache
# ============================================================

@dataclass
class NodeInfo:
    value: float                 # cp, side to move
    result: float                # expected result in [-1, 1], side to move
    moves: list[chess.Move]      # legal moves
    priors: np.ndarray           # policy probability for each legal move


class NeuralEvaluator:
    """Runs the network on positions and caches what it learns about them."""

    def __init__(self, model: torch.nn.Module, device: str | torch.device = "cpu",
                 cache_size: int = 300_000):
        self.model = model
        self.device = torch.device(device)
        self.cache: OrderedDict[int, NodeInfo] = OrderedDict()
        self.cache_size = cache_size
        self.evals = 0

    def clear(self):
        self.cache.clear()

    def get(self, key: int) -> NodeInfo | None:
        return self.cache.get(key)

    def evaluate(self, boards: list[chess.Board], keys: list[int]) -> list[NodeInfo]:
        """Evaluate positions not already cached, in a single forward pass."""
        out: list[NodeInfo | None] = [self.cache.get(k) for k in keys]
        todo = [i for i, info in enumerate(out) if info is None]

        if todo:
            planes = np.stack([board_to_planes(boards[i]) for i in todo])
            x = torch.from_numpy(planes).to(self.device)
            with torch.inference_mode():
                logits, _, _, cp, _, result = self.model(x)
            logits = logits.float().cpu().numpy()
            cp = cp.float().cpu().numpy()
            result = result.float().cpu().numpy()
            self.evals += len(todo)

            for row, i in enumerate(todo):
                moves = list(boards[i].legal_moves)
                if moves:
                    lg = logits[row, [move_to_index(m) for m in moves]].astype(np.float64)
                    lg = np.exp(lg - lg.max())
                    priors = lg / lg.sum()
                else:
                    priors = np.zeros(0)
                info = NodeInfo(float(cp[row]), float(result[row]), moves, priors)
                out[i] = info
                self.cache[keys[i]] = info

            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)

        return out  # type: ignore[return-value]


# ============================================================
# Search
# ============================================================

@dataclass
class SearchConfig:
    time_limit: float = 1.0      # seconds; None for no limit
    max_depth: int = 8
    multipv: int = 3
    batch_size: int = 16
    prune_width: int = 8         # moves kept per interior node (plus tactics)
    prune_mass: float = 0.95     # ...or until this much prior mass is covered
    qsearch_depth: int = 4       # max plies of captures resolved at the leaves
    tt_size: int = 500_000


@dataclass
class _TTEntry:
    depth: int
    score: float
    flag: int
    move: chess.Move | None


class Searcher:
    def __init__(self, evaluator: NeuralEvaluator, config: SearchConfig | None = None):
        self.ev = evaluator
        self.cfg = config or SearchConfig()
        self.tt: OrderedDict[int, _TTEntry] = OrderedDict()
        self.nodes = 0
        self._deadline: float | None = None

    def reset(self):
        self.tt.clear()
        self.ev.clear()

    # ---------------- helpers ----------------

    @staticmethod
    def _key(board: chess.Board):
        return position_key(board)

    def _check_time(self):
        if self._deadline is not None and time.perf_counter() > self._deadline:
            raise SearchTimeout

    @staticmethod
    def _to_tt(score: float, ply: int) -> float:
        # Mate scores are stored relative to the node, not the root.
        if score >= MATE_THRESHOLD:
            return score + ply
        if score <= -MATE_THRESHOLD:
            return score - ply
        return score

    @staticmethod
    def _from_tt(score: float, ply: int) -> float:
        if score >= MATE_THRESHOLD:
            return score - ply
        if score <= -MATE_THRESHOLD:
            return score + ply
        return score

    def _tt_store(self, key: int, depth: int, score: float, flag: int, move):
        old = self.tt.get(key)
        if old is None or depth >= old.depth:
            self.tt[key] = _TTEntry(depth, score, flag, move)
            if len(self.tt) > self.cfg.tt_size:
                self.tt.popitem(last=False)

    @staticmethod
    def _is_draw(board: chess.Board) -> bool:
        return (
            board.halfmove_clock >= 100
            or board.is_insufficient_material()
            or board.is_repetition(2)
        )

    def _ordered_moves(self, board: chess.Board, info: NodeInfo, tt_move, prune: bool):
        order = np.argsort(-info.priors)
        moves = [info.moves[i] for i in order]
        priors = [float(info.priors[i]) for i in order]

        if tt_move is not None and tt_move in info.moves:
            idx = moves.index(tt_move)
            moves.insert(0, moves.pop(idx))
            priors.insert(0, priors.pop(idx))

        if not prune or board.is_check():
            return moves

        kept, mass = [], 0.0
        for mv, p in zip(moves, priors):
            within_budget = len(kept) < self.cfg.prune_width and mass < self.cfg.prune_mass
            if (
                within_budget
                or mv == tt_move
                or mv.promotion is not None
                or board.is_capture(mv)
                or board.gives_check(mv)
            ):
                kept.append(mv)
                mass += p
        return kept

    @staticmethod
    def _noisy_moves(board: chess.Board, moves: list[chess.Move]) -> list[tuple[int, chess.Move]]:
        """
        Captures and queen promotions worth looking at in quiescence, as
        (material gain, move), most valuable victim first. A capture is kept
        if it wins at least as much as the capturing piece is worth, or if
        the target square is not defended.
        """
        them = not board.turn
        out = []
        for mv in moves:
            capture = board.is_capture(mv)
            if not capture and mv.promotion != chess.QUEEN:
                continue
            attacker = PIECE_VALUE[board.piece_type_at(mv.from_square)]
            victim = PIECE_VALUE[board.piece_type_at(mv.to_square) or chess.PAWN] if capture else 0
            gain = victim + (PIECE_VALUE[mv.promotion] - 100 if mv.promotion else 0)
            if mv.promotion or victim >= attacker or not board.is_attacked_by(them, mv.to_square):
                out.append((gain, mv))
        out.sort(key=lambda t: t[0], reverse=True)
        return out

    def _prefetch(self, board: chess.Board, moves: list[chess.Move]):
        """Evaluate the children of a frontier node in one batch."""
        boards, keys = [], []
        for mv in moves:
            board.push(mv)
            key = self._key(board)
            if self.ev.get(key) is None:
                boards.append(board.copy(stack=False))
                keys.append(key)
            board.pop()
        if boards:
            self.ev.evaluate(boards, keys)

    # ---------------- negamax ----------------

    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int) -> float:
        self.nodes += 1
        if self.nodes & 15 == 0:
            self._check_time()

        if self._is_draw(board):
            return 0.0

        key = self._key(board)
        info = self.ev.get(key) or self.ev.evaluate([board], [key])[0]

        if not info.moves:
            return -(MATE_SCORE - ply) if board.is_check() else 0.0
        if depth <= 0:
            return self._qsearch(board, alpha, beta, ply, self.cfg.qsearch_depth, info)

        alpha0 = alpha
        entry = self.tt.get(key)
        tt_move = None
        if entry is not None:
            tt_move = entry.move
            if entry.depth >= depth:
                tt_score = self._from_tt(entry.score, ply)
                if entry.flag == EXACT:
                    return tt_score
                if entry.flag == LOWER:
                    alpha = max(alpha, tt_score)
                elif entry.flag == UPPER:
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score

        moves = self._ordered_moves(board, info, tt_move, prune=True)

        best, best_move = -INF, None
        for i, mv in enumerate(moves):
            if depth == 1 and i % self.cfg.batch_size == 0:
                self._prefetch(board, moves[i:i + self.cfg.batch_size])

            board.push(mv)
            try:
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()

            if score > best:
                best, best_move = score, mv
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        flag = UPPER if best <= alpha0 else LOWER if best >= beta else EXACT
        self._tt_store(key, depth, self._to_tt(best, ply), flag, best_move)
        return best

    def _qsearch(self, board: chess.Board, alpha: float, beta: float, ply: int,
                 qdepth: int, info: NodeInfo | None = None) -> float:
        """Resolve captures (and check evasions) before trusting the network."""
        if info is None:
            self.nodes += 1
            if self.nodes & 15 == 0:
                self._check_time()
            key = self._key(board)
            info = self.ev.get(key) or self.ev.evaluate([board], [key])[0]
            if not info.moves:
                return -(MATE_SCORE - ply) if board.is_check() else 0.0

        if qdepth <= 0:
            return info.value

        if board.is_check():
            # No standing pat while in check: every evasion has to be tried.
            best = -INF
            moves = [info.moves[i] for i in np.argsort(-info.priors)]
        else:
            stand = info.value
            if stand >= beta:
                return stand
            best = stand
            alpha = max(alpha, stand)
            moves = [mv for gain, mv in self._noisy_moves(board, info.moves)
                     if stand + gain + DELTA_MARGIN > alpha]
            if not moves:
                return stand

        self._prefetch(board, moves[: self.cfg.batch_size])
        for mv in moves:
            board.push(mv)
            try:
                score = -self._qsearch(board, -beta, -alpha, ply + 1, qdepth - 1)
            finally:
                board.pop()
            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        return best

    def _principal_variation(self, board: chess.Board, first: chess.Move, max_len: int) -> list[chess.Move]:
        pv = [first]
        b = board.copy()
        b.push(first)
        seen = {self._key(b)}
        while len(pv) < max_len:
            entry = self.tt.get(self._key(b))
            if entry is None or entry.move is None or entry.move not in b.legal_moves:
                break
            pv.append(entry.move)
            b.push(entry.move)
            k = self._key(b)
            if k in seen:
                break
            seen.add(k)
        return pv

    # ---------------- root ----------------

    def _search_root(self, board: chess.Board, depth: int, root_moves: list[chess.Move],
                     priors: dict[chess.Move, float]) -> list[Line]:
        """One iteration of MultiPV search at a fixed depth."""
        k = max(1, self.cfg.multipv)
        scored: list[tuple[float, chess.Move, bool]] = []

        if depth == 1:
            self._prefetch(board, root_moves)

        for mv in root_moves:
            exact_scores = sorted((s for s, _, e in scored if e), reverse=True)
            alpha = exact_scores[k - 1] if len(exact_scores) >= k else -INF

            board.push(mv)
            try:
                score = -self._negamax(board, depth - 1, -INF, -alpha, 1)
            finally:
                board.pop()

            scored.append((score, mv, score > alpha or alpha == -INF))

        scored.sort(key=lambda t: (t[2], t[0]), reverse=True)
        return [
            Line(mv, score, self._principal_variation(board, mv, depth + 4) if exact else [mv],
                 priors.get(mv, 0.0), exact)
            for score, mv, exact in scored
        ]

    def search(self, board: chess.Board, time_limit: float | None = None,
               max_depth: int | None = None) -> SearchResult:
        cfg = self.cfg
        time_limit = cfg.time_limit if time_limit is None else time_limit
        max_depth = max_depth or cfg.max_depth

        start = time.perf_counter()
        self.nodes = 0
        evals_before = self.ev.evals
        board = board.copy()
        # Scores near repetitions depend on the game history, so the tree is
        # not reused between searches. Network evaluations are pure functions
        # of the position and stay cached.
        self.tt.clear()

        root_key = self._key(board)
        root = self.ev.get(root_key) or self.ev.evaluate([board], [root_key])[0]
        priors = {mv: float(p) for mv, p in zip(root.moves, root.priors)}

        if not root.moves:
            score = -MATE_SCORE if board.is_check() else 0.0
            return SearchResult(None, score, 0, 1, self.ev.evals - evals_before,
                                (time.perf_counter() - start) * 1000, policy=priors,
                                static_eval=root.value, win_prob=(root.result + 1) / 2)

        root_moves = sorted(root.moves, key=lambda m: -priors[m])
        lines: list[Line] = []
        completed = 0

        for depth in range(1, max_depth + 1):
            # Depth 1 always runs to completion so there is always an answer.
            self._deadline = None if depth == 1 or time_limit is None else start + time_limit
            try:
                lines = self._search_root(board, depth, root_moves, priors)
                completed = depth
            except SearchTimeout:
                break

            root_moves = [l.move for l in lines]
            if abs(lines[0].score) >= MATE_THRESHOLD:
                break  # forced mate found; deeper search will not change the move
            if time_limit is not None:
                elapsed = time.perf_counter() - start
                # The next iteration usually costs several times the last one.
                if elapsed > time_limit * 0.4:
                    break

        self._deadline = None
        best = lines[0]
        return SearchResult(
            best_move=best.move,
            score=best.score,
            depth=completed,
            nodes=self.nodes,
            nn_evals=self.ev.evals - evals_before,
            time_ms=(time.perf_counter() - start) * 1000,
            pv=best.pv,
            lines=lines,
            policy=priors,
            static_eval=root.value,
            win_prob=(root.result + 1) / 2,
        )


    def instinct(self, board: chess.Board) -> SearchResult:
        """The network's raw choice: one forward pass, no search."""
        start = time.perf_counter()
        evals_before = self.ev.evals
        key = self._key(board)
        info = self.ev.get(key) or self.ev.evaluate([board], [key])[0]
        policy = {mv: float(p) for mv, p in zip(info.moves, info.priors)}
        ranked = sorted(policy, key=policy.get, reverse=True)
        lines = [Line(mv, info.value, [mv], policy[mv]) for mv in ranked[: self.cfg.multipv]]
        return SearchResult(
            best_move=ranked[0] if ranked else None,
            score=info.value,
            depth=0,
            nodes=1,
            nn_evals=self.ev.evals - evals_before,
            time_ms=(time.perf_counter() - start) * 1000,
            pv=ranked[:1],
            lines=lines,
            policy=policy,
            static_eval=info.value,
            win_prob=(info.result + 1) / 2,
        )


# ============================================================
# Convenience
# ============================================================

def win_probability(score_cp: float) -> float:
    """Logistic mapping from centipawns to an expected score in [0, 1]."""
    if score_cp >= MATE_THRESHOLD:
        return 1.0
    if score_cp <= -MATE_THRESHOLD:
        return 0.0
    return 1.0 / (1.0 + math.exp(-0.00368208 * score_cp))
