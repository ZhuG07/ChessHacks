#!/usr/bin/env python3
"""
SELF-PLAY DATA GENERATOR (STOCKFISH-LABELED)
============================================

Produces positions with:
  - X: planes (18, 8, 8)
  - y_policy_best: Stockfish best-move index
  - cp_before: eval of current position (stm POV)
  - cp_after_best: eval after best move (same POV)
  - delta_cp: cp_after_best - cp_before
  - game_result: final game result (POV of side to move at each sample)

Every stored position is produced by:

  1) Get SF best move + eval
  2) Optionally filter by game phase (favor midgame/endgame)
  3) Store labels
  4) Choose *noisy move* for self-play:
       - temperature
       - Dirichlet noise

Outputs:
  shard_00001.npz
  shard_00002.npz
  ...

Each shard contains ~SHARD_SIZE positions.
"""

import chess
import chess.engine
import numpy as np
from pathlib import Path
import random

# =====================================================================
# CONFIG
# =====================================================================

class Config:
    ENGINE_PATH = r"F:/VS Code Storage/ChessHacks2025/training\whiteNoise\stockfish-windows-x86-64-avx2.exe"

    DEPTH = 12                  # use fixed depth for reproducibility
    TIME_LIMIT = None           # or e.g. 0.03 for time-based

    MAX_MOVES_PER_GAME = 200

    SHARD_SIZE = 50_000         # NPZ entries per shard
    OUTPUT_DIR = r"training\whiteNoise/processed"

    DIRICHLET_ALPHA = 0.3
    TEMPERATURE = 1.2           # >1 = more randomness

    # -------- Phase-based filtering (0=endgame, 1=opening) --------
    OPENING_PHASE_THRESHOLD = 0.7  # above this is "opening-ish"
    ENDGAME_PHASE_THRESHOLD = 0.3  # below this is "endgame-ish"

    # Keep only this fraction of opening positions (others are skipped,
    # but the game continues with self-play).
    OPENING_KEEP_PROB = 0.3        # keep 30% of opening positions

    # Optionally drop totally lopsided positions (in cp) outside pure endgames
    MAX_ABS_CP_FOR_STORAGE = 800   # skip |cp_before| > this, unless in endgame

cfg = Config()  


# =====================================================================
# BOARD → PLANES (same format as your supervised pipeline)
# =====================================================================

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
NUM_PLANES = 18

PROMO_PIECES = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
NUM_PROMOS = len(PROMO_PIECES)
POLICY_DIM = 64 * 64 * NUM_PROMOS


def move_to_index(move: chess.Move) -> int:
    from_sq = move.from_square
    to_sq = move.to_square
    promo = move.promotion if move.promotion is not None else None
    promo_idx = PROMO_PIECES.index(promo)
    return (from_sq * 64 + to_sq) * NUM_PROMOS + promo_idx


def fen_to_planes(board: chess.Board) -> np.ndarray:
    P = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for sq, piece in board.piece_map().items():
        p_idx = PIECE_PLANES[(piece.piece_type, piece.color)]
        r = 7 - chess.square_rank(sq)
        f = chess.square_file(sq)
        P[p_idx, r, f] = 1.0

    # side to move
    P[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # castling
    if board.has_kingside_castling_rights(chess.WHITE):
        P[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        P[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        P[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        P[16, :, :] = 1.0

    # en passant
    if board.ep_square is not None:
        file = chess.square_file(board.ep_square)
        P[17, :, file] = 1.0

    return P


# =====================================================================
# GAME PHASE ESTIMATION (material-based)
# =====================================================================

# Rough phase values per piece (similar spirit to Stockfish)
PIECE_PHASE = {
    chess.PAWN:   0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK:   2,
    chess.QUEEN:  4,
    chess.KING:   0,
}

# Max phase for full material: 2 knights, 2 bishops, 2 rooks, 1 queen per side
MAX_PHASE = (
    2 * PIECE_PHASE[chess.KNIGHT] * 2 +  # 2 knights per side
    2 * PIECE_PHASE[chess.BISHOP] * 2 +  # 2 bishops per side
    2 * PIECE_PHASE[chess.ROOK]   * 2 +  # 2 rooks per side
    2 * PIECE_PHASE[chess.QUEEN]  * 1    # 1 queen per side
)


def game_phase_from_board(board: chess.Board) -> float:
    """
    Return phase in [0,1]:
      1.0 ~ full material / opening-ish
      0.0 ~ bare material / pure endgame-ish
    """
    phase = 0
    for piece_type, p_val in PIECE_PHASE.items():
        if p_val == 0:
            continue
        for color in [chess.WHITE, chess.BLACK]:
            count = len(board.pieces(piece_type, color))
            phase += count * p_val

    phase = min(phase, MAX_PHASE) if MAX_PHASE > 0 else 0
    return phase / MAX_PHASE if MAX_PHASE > 0 else 0.0


# =====================================================================
# STOCKFISH WRAPPER
# =====================================================================

class StockfishEvaluator:
    def __init__(self, path, depth=None, time_limit=None):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.depth = depth
        self.time_limit = time_limit

    def limit(self):
        if self.depth is not None:
            return chess.engine.Limit(depth=self.depth)
        return chess.engine.Limit(time=self.time_limit)

    def eval_and_best(self, board: chess.Board):
        """
        Return (cp, best_move) from POV of side to move.
        cp is centipawns, mate mapped to +/-10000.
        """
        info = self.engine.analyse(board, self.limit())
        score = info["score"].pov(board.turn)
        if score.is_mate():
            mate = score.mate()
            cp = 10000 * (1 if mate > 0 else -1)
        else:
            cp = score.score(mate_score=10000)

        pv = info.get("pv", None)
        best_move = pv[0] if pv else None
        return int(cp), best_move

    def close(self):
        self.engine.quit()


# =====================================================================
# SELF-PLAY EPISODE
# =====================================================================

def choose_random_move(board: chess.Board) -> chess.Move:
    """
    Returns a random legal move. We can bias slightly toward best move if desired.
    """
    legal = list(board.legal_moves)
    if not legal:
        return None
    return random.choice(legal)


def play_selfplay_game_random(sf: StockfishEvaluator):
    """
    Generates positions by playing random moves (not necessarily best).
    Labels are generated by evaluating the move using Stockfish.
    """
    board = chess.Board()
    labels = []

    while not board.is_game_over() and len(labels) < cfg.MAX_MOVES_PER_GAME:
        # Evaluate current position
        cp_before, _ = sf.eval_and_best(board)  # best move not needed here

        # Choose a random move for self-play
        move = choose_random_move(board)
        if move is None:
            break

        # Push move
        board.push(move)

        # Evaluate after move
        cp_after, best_after_move = sf.eval_and_best(board)

        # Delta from perspective of side to move before the move
        delta = cp_after - cp_before

        # Policy target is always the move played
        policy_idx = move_to_index(move)

        # Store planes + labels
        planes = fen_to_planes(board)
        labels.append((planes, policy_idx, cp_before, cp_after, delta))

    # Convert final game result
    res = board.result()
    if res == "1-0":
        g = 1.0
    elif res == "0-1":
        g = -1.0
    else:
        g = 0.0

    return labels, g



# =====================================================================
# MAIN LOOP — WRITES SHARDED NPZ FILES
# =====================================================================

def main():
    out = Path(cfg.OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    sf = StockfishEvaluator(cfg.ENGINE_PATH, depth=cfg.DEPTH, time_limit=cfg.TIME_LIMIT)

    shard_id = 1 #MANUALLY CHANGE EVERYTIME TO AVOID OVERWRITING
    buffer = []

    print("Starting self-play...")

    try:
        while True:
            labels, g = play_selfplay_game_random(sf)

            # Attach game result g to each position
            for (planes, pol, cp_before, cp_after, delta) in labels:
                buffer.append((planes, pol, cp_before, cp_after, delta, g))

            if len(buffer) >= cfg.SHARD_SIZE:
                X = np.stack([b[0] for b in buffer])
                policy = np.array([b[1] for b in buffer], np.int64)
                cp_before = np.array([b[2] for b in buffer], np.float32)
                cp_after = np.array([b[3] for b in buffer], np.float32)
                delta_cp = np.array([b[4] for b in buffer], np.float32)
                game_result = np.array([b[5] for b in buffer], np.float32)

                save_path = out / f"gz_shard_{shard_id:05d}.npz"
                np.savez_compressed(
                    save_path,
                    X=X,
                    y_policy_best=policy,
                    cp_before=cp_before,
                    cp_after_best=cp_after,
                    delta_cp=delta_cp,
                    game_result=game_result,
                )

                print(f"[+] Saved shard {shard_id} with {len(buffer)} samples → {save_path}")

                buffer = []
                shard_id += 1

            # (optional) small heartbeat print
            if shard_id % 10 == 0:
                print(f"Still running… shards={shard_id-1}")

    finally:
        sf.close()


if __name__ == "__main__":
    main()
