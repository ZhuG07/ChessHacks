from .utils import chess_manager, GameContext
from chess import Move
import chess
import random
import time
from pathlib import Path

import numpy as np
import torch

# ============================================================
# CONFIG
# ============================================================

# Relative path: src/main.py → src/model_save/best.pt
MODEL_PATH = Path(__file__).resolve().parent / "model_save" / "best.pt"

NUM_PLANES = 18
NUM_PROMOS = 5   # [None, Q, R, B, N]
POLICY_DIM = 64 * 64 * NUM_PROMOS
CP_SCALE = 200.0  # must match training

MATE_SCORE = 100000.0  # not really used here, kept for future extensions

ENGINE = None  # will be set if NN loads


# ============================================================
# BOARD → PLANES (must match training)
# ============================================================

PIECE_PLANES = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}

PROMO_PIECES = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]


def board_to_planes(board: chess.Board) -> np.ndarray:
    """
    Same encoding as training: (18, 8, 8).
    """
    P = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for sq, piece in board.piece_map().items():
        p_idx = PIECE_PLANES[(piece.piece_type, piece.color)]
        r = 7 - chess.square_rank(sq)
        f = chess.square_file(sq)
        P[p_idx, r, f] = 1.0

    # side to move
    if board.turn == chess.WHITE:
        P[12, :, :] = 1.0
    else:
        P[12, :, :] = 0.0

    # castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        P[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        P[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        P[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        P[16, :, :] = 1.0

    # en-passant file
    if board.ep_square is not None:
        file = chess.square_file(board.ep_square)
        P[17, :, file] = 1.0

    return P


def move_to_index(move: chess.Move) -> int:
    """
    Same policy index scheme as training:
      (from * 64 + to) * 5 + promo_idx
    """
    from_sq = move.from_square
    to_sq = move.to_square
    promo = move.promotion if move.promotion is not None else None
    promo_idx = PROMO_PIECES.index(promo)  # 0..4
    return (from_sq * 64 + to_sq) * NUM_PROMOS + promo_idx


# ============================================================
# Simple NN policy engine (no search, 1 forward per move)
# ============================================================

try:
    from .model import ResNet20_PolicyDelta
except Exception as e:
    ResNet20_PolicyDelta = None
    print(f"[ENGINE] Could not import ResNet20_PolicyDelta from .model: {e}")


class PolicyOnlyEngine:
    """
    Minimal NN engine:
      - Loads ResNet20_PolicyDelta
      - Uses policy head to score legal moves
      - Samples / chooses moves from softmax over legal moves
    """

    def __init__(self):
        if ResNet20_PolicyDelta is None:
            raise RuntimeError("Model class not available")

        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ENGINE] Using device: {self.device}")

        self.model = ResNet20_PolicyDelta(
            in_planes=NUM_PLANES,
            policy_dim=POLICY_DIM,
            cp_scale=CP_SCALE,
        ).to(self.device)

        state = torch.load(MODEL_PATH, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

        print(f"[ENGINE] Loaded model from {MODEL_PATH}")

    @torch.no_grad()
    def _policy_logits(self, board: chess.Board) -> np.ndarray:
        """
        Run NN once and return policy logits as a NumPy array of shape (POLICY_DIM,).
        """
        planes = board_to_planes(board)  # (18, 8, 8)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)  # (1,18,8,8)

        # Unified model signature: logits, delta_real, delta_pred, cp_real, cp_pred, res_pred
        logits, _, _, _, _, _ = self.model(x)
        logits = logits[0].detach().cpu().numpy()  # (POLICY_DIM,)
        return logits

    @torch.no_grad()
    def select_move(self, board: chess.Board):
        """
        Compute softmax over NN policy for the current legal moves and pick one.
        Returns (best_move, move_prob_dict).
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, {}

        logits = self._policy_logits(board)
        scores = []

        for mv in legal_moves:
            idx = move_to_index(mv)
            if 0 <= idx < POLICY_DIM:
                scores.append(float(logits[idx]))
            else:
                # If something falls outside range, give it a very low score
                scores.append(-1e9)

        # Softmax over legal moves
        max_s = max(scores)
        exps = [np.exp(s - max_s) if s > -1e8 else 0.0 for s in scores]
        Z = float(sum(exps))

        if Z <= 0.0:
            # Fallback: uniform if something goes terribly wrong
            p = 1.0 / len(legal_moves)
            move_probs = {mv: p for mv in legal_moves}
            best_move = random.choice(legal_moves)
            return best_move, move_probs

        probs = [e / Z for e in exps]
        move_probs = {mv: p for mv, p in zip(legal_moves, probs)}

        # Choose argmax (or sample for more randomness)
        best_idx = int(np.argmax(probs))
        best_move = legal_moves[best_idx]

        return best_move, move_probs

    def reset(self):
        # No per-game state right now, but hook kept for future
        pass


# Try to construct the engine once at import
try:
    if ResNet20_PolicyDelta is not None:
        ENGINE = PolicyOnlyEngine()
    else:
        ENGINE = None
except Exception as e:
    ENGINE = None
    print(f"[ENGINE] Failed to initialize NN engine, falling back to random. Error: {e}")


# ============================================================
# Submission platform hooks
# ============================================================

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Cooking move...")
    print(ctx.board.move_stack)
    time.sleep(0.01)  # small delay; can be reduced/removed

    board = ctx.board
    legal_moves = list(board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    # If NN engine didn't initialize, fall back to random baseline
    if ENGINE is None:
        print("Safety")
        move_weights = [random.random() for _ in legal_moves]
        total_weight = sum(move_weights)
        move_probs = {
            move: weight / total_weight
            for move, weight in zip(legal_moves, move_weights)
        }
        ctx.logProbabilities(move_probs)
        return random.choices(legal_moves, weights=move_weights, k=1)[0]

    # Otherwise, NN policy for move selection
    best_move, move_probs = ENGINE.select_move(board)

    if best_move is None:
        # Extra safety fallback
        print("Safety")
        move_weights = [random.random() for _ in legal_moves]
        total_weight = sum(move_weights)
        move_probs = {
            move: weight / total_weight
            for move, weight in zip(legal_moves, move_weights)
        }
        ctx.logProbabilities(move_probs)
        return random.choices(legal_moves, weights=move_weights, k=1)[0]

    # Ensure probabilities are defined over exactly current legal moves
    probs_filtered = {mv: move_probs.get(mv, 0.0) for mv in legal_moves}
    Z = sum(probs_filtered.values())
    if Z > 0:
        probs_filtered = {mv: p / Z for mv, p in probs_filtered.items()}
    else:
        # Degenerate case → uniform
        p = 1.0 / len(legal_moves)
        probs_filtered = {mv: p for mv in legal_moves}

    ctx.logProbabilities(probs_filtered)
    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    if ENGINE is not None:
        ENGINE.reset()
