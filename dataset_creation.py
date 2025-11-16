import chess
import chess.pgn
import chess.engine
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import random

# ---------- CONFIG: EDIT THESE ----------
# Your big PGN file:
PGN_PATH = Path(r"training\utils\LumbrasGigaBase_OTB_2015-2019.pgn")#OAWHDOIAWHDOOHAWHDOIHOIWOAHD

# Where to save the dataset:
OUT_PATH = Path(r"C:\Users\ethan\Downloads\ChessHacks\e\ChessHacks\my-chesshacks-bot\processed")

# Path to Stockfish binary:
ENGINE_PATH = r"C:\Users\ethan\Downloads\ChessHacks\e\ChessHacks\my-chesshacks-bot\src\stockfish-windows-x86-64-avx2.exe"  # fix slashes or use raw string

SAMPLE_EVERY = 1      # 1 = every ply, 2 = every second ply, etc.
MAX_GAMES = 400         # or None for all games
TIME_LIMIT = 0.05     # seconds per position if DEPTH is None
DEPTH = 14            # e.g. 16 for fixed depth instead of time

# Debug controls:
DEBUG = False         # set True for verbose per-game/per-ply debug
DEBUG_GAMES = 1       # how many games to print debug info for
DEBUG_PLY_LIMIT = 5   # how many plies per debug game to print
POSTHOC_DEBUG_SAMPLES = 5  # how many random samples to inspect at the end
# ----------------------------------------


# ---------- Stockfish wrapper ----------
class StockfishEvaluator:
    """
    Simple wrapper around Stockfish for supervised labeling.
    Provides both evaluation and best move from a single call.
    """

    def __init__(self, engine_path: str, time_limit: float = 0.1, depth: Optional[int] = None):
        self.engine_path = str(engine_path)
        self.engine = chess.engine.SimpleEngine.popen_uci(Path(engine_path))
        self.time_limit = time_limit
        self.depth = depth

    def _make_limit(self) -> chess.engine.Limit:
        if self.depth is not None:
            return chess.engine.Limit(depth=self.depth)
        return chess.engine.Limit(time=self.time_limit)

    def eval_and_best(self, board: chess.Board) -> Tuple[int, Optional[chess.Move]]:
        """
        Return (centipawn eval, best move) from the perspective of the side to move.
        Positive cp = advantage for side to move.
        Also returns a principal-variation best move.
        """
        limit = self._make_limit()
        info = self.engine.analyse(board, limit)  # single PV

        score = info["score"].pov(board.turn)

        if score.is_mate():
            mate_in = score.mate()
            sign = 1 if mate_in is None or mate_in > 0 else -1
            cp = 10000 * sign
        else:
            cp = score.score(mate_score=10000)

        pv = info.get("pv", None)
        best_move = pv[0] if pv else None

        return int(cp), best_move

    def evaluate_cp(self, board: chess.Board) -> int:
        """
        Return centipawn eval from the perspective of the side to move.
        Positive means advantage for side to move.
        """
        limit = self._make_limit()
        info = self.engine.analyse(board, limit)
        score = info["score"].pov(board.turn)

        if score.is_mate():
            mate_in = score.mate()
            sign = 1 if mate_in is None or mate_in > 0 else -1
            cp = 10000 * sign
        else:
            cp = score.score(mate_score=10000)
        return int(cp)

    def close(self):
        self.engine.quit()


# ---------- Board encoding (FEN -> planes) ----------
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
NUM_PLANES = 18  # 12 pieces + 1 stm + 4 castling + 1 ep-file


def fen_to_planes(fen: str) -> np.ndarray:
    board = chess.Board(fen)
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        p_idx = PIECE_PLANES[(piece.piece_type, piece.color)]
        rank = 7 - chess.square_rank(square)
        file = chess.square_file(square)
        planes[p_idx, rank, file] = 1.0

    # side to move
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # castling
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16, :, :] = 1.0

    # en passant file
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        planes[17, :, ep_file] = 1.0

    return planes


# ---------- Move encoding (move <-> index) ----------
PROMO_PIECES = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
NUM_PROMOS = len(PROMO_PIECES)
POLICY_DIM = 64 * 64 * NUM_PROMOS  # 20,480


def encode_move_components(from_sq: int, to_sq: int, promo_piece) -> int:
    promo_idx = PROMO_PIECES.index(promo_piece)
    return (from_sq * 64 + to_sq) * NUM_PROMOS + promo_idx


def move_to_index(move: chess.Move) -> int:
    from_sq = move.from_square
    to_sq = move.to_square
    promo_piece = move.promotion if move.promotion is not None else None
    return encode_move_components(from_sq, to_sq, promo_piece)


# ---------- Helpers for results ----------
def pgn_result_to_outcome_white(result_str: str) -> Optional[float]:
    if result_str == "1-0":
        return 1.0
    elif result_str == "0-1":
        return -1.0
    elif result_str == "1/2-1/2":
        return 0.0
    else:
        return None


def outcome_from_side_to_move(fen: str, outcome_white: float) -> float:
    """
    Convert game result from White's POV to side-to-move POV
    for the given FEN.
    """
    board = chess.Board(fen)
    if outcome_white == 0.0:
        return 0.0
    if board.turn == chess.WHITE:
        return outcome_white
    else:
        return -outcome_white


# ---------- Main dataset builder with debug ----------
def main():
    if not PGN_PATH.exists():
        raise FileNotFoundError(f"PGN file not found: {PGN_PATH}")

    print(f"Using PGN: {PGN_PATH}")
    print(f"Stockfish: {ENGINE_PATH}")
    print(f"Output   : {OUT_PATH}")
    print(f"sample_every={SAMPLE_EVERY}, max_games={MAX_GAMES}, "
          f"time_limit={TIME_LIMIT}, depth={DEPTH}")
    print(f"Policy dimension = {POLICY_DIM}")

    sf = StockfishEvaluator(ENGINE_PATH, time_limit=TIME_LIMIT, depth=DEPTH)

    X_list: list[np.ndarray] = []
    y_policy_best_list: list[int] = []
    cp_before_list: list[float] = []
    cp_after_best_list: list[float] = []
    delta_cp_list: list[float] = []
    human_move_idx_list: list[int] = []
    human_delta_list: list[float] = []
    game_result_list: list[float] = []

    debug_games_printed = 0

    game_count = 0
    with open(PGN_PATH, encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_count += 1
            if MAX_GAMES is not None and game_count > MAX_GAMES:
                break

            result_str = game.headers.get("Result", "*")
            outcome_white = pgn_result_to_outcome_white(result_str)
            if outcome_white is None:
                continue  # skip weird/unfinished games

            board = game.board()

            # Skip non-standard initial positions (like knight odds)
            if board.fen().split(" ")[0] != chess.STARTING_FEN.split(" ")[0]:
                if DEBUG:
                    print(f"Skipping game {game_count}: non-standard initial FEN:", board.fen())
                    continue  # <- always skip, not only in DEBUG

            ply_idx = 0

            # Debug: print basic game info for first few games
            is_debug_game = DEBUG and debug_games_printed < DEBUG_GAMES
            if is_debug_game:
                debug_games_printed += 1
                print("\n========== DEBUG GAME", game_count, "==========")
                print("Event :", game.headers.get("Event", "?"))
                print("White :", game.headers.get("White", "?"))
                print("Black :", game.headers.get("Black", "?"))
                print("Result:", result_str)
                print("Initial FEN:", board.fen())
                print("====================================")

            debug_plies_printed = 0

            for move in game.mainline_moves():
                fen_before = board.fen()

                if ply_idx % SAMPLE_EVERY == 0:
                    # Make sure PGN move is legal in this position
                    if move not in board.legal_moves:
                        if DEBUG:
                            print("Warning: PGN move not legal; skipping position.")
                        board.push(move)
                        ply_idx += 1
                        continue

                    planes = fen_to_planes(fen_before)

                    try:
                        # 1) Eval & best move from this position (stm POV)
                        cp_before_stm, best_move = sf.eval_and_best(board)
                    except Exception as e:
                        print(f"Stockfish error (eval_and_best), skipping position: {e}")
                        board.push(move)
                        ply_idx += 1
                        continue

                    if best_move is None:
                        if DEBUG:
                            print("No best move returned by Stockfish; skipping position.")
                        board.push(move)
                        ply_idx += 1
                        continue

                    # 2) Eval after Stockfish best move, from POV of original side-to-move
                    try:
                        board_best = board.copy()
                        board_best.push(best_move)
                        cp_after_best_stm = sf.evaluate_cp(board_best)
                        # cp_before_stm: original side-to-move POV at s
                        # cp_after_best_stm: POV of the opponent at s'
                        # -> original side POV at s' is the negation
                        cp_after_best = -float(cp_after_best_stm)
                    except Exception as e:
                        print(f"Stockfish error (cp_after_best), skipping position: {e}")
                        board.push(move)
                        ply_idx += 1
                        continue

                    # 3) Human move index and its eval delta (optional)
                    human_move_idx = move_to_index(move)
                    try:
                        board_human = board.copy()
                        board_human.push(move)
                        cp_after_human_stm = sf.evaluate_cp(board_human)
                        cp_after_human = -float(cp_after_human_stm)  # original side POV
                        human_delta = cp_after_human - float(cp_before_stm)
                    except Exception as e:
                        print(f"Stockfish error (cp_after_human), marking human_delta=0: {e}")
                        cp_after_human = float("nan")
                        human_delta = 0.0
                        human_move_idx = -1  # mark as invalid

                    cp_before = float(cp_before_stm)
                    delta_cp = cp_after_best - cp_before

                    z = outcome_from_side_to_move(fen_before, float(outcome_white))

                    # Append all labels
                    X_list.append(planes)
                    y_policy_best_list.append(move_to_index(best_move))
                    cp_before_list.append(cp_before)
                    cp_after_best_list.append(cp_after_best)
                    delta_cp_list.append(delta_cp)
                    human_move_idx_list.append(human_move_idx)
                    human_delta_list.append(human_delta)
                    game_result_list.append(z)

                    # Debug: log first few plies of first few games
                    if is_debug_game and debug_plies_printed < DEBUG_PLY_LIMIT:
                        san_human = board.san(move)
                        san_best = board.san(best_move)
                        print(f"\n[DEBUG] Game {game_count}, ply {ply_idx}")
                        print("Side to move:",
                              "White" if board.turn == chess.WHITE else "Black")
                        print("FEN before move:")
                        print(fen_before)
                        print("Human move (SAN):", san_human)
                        print("SF best move (SAN):", san_best)
                        print("cp_before (stm POV)              :", cp_before)
                        print("cp_after_best_stm (opp POV at s') :", cp_after_best_stm)
                        print("cp_after_best (orig side POV)     :", cp_after_best)
                        print("delta_cp (best - before)          :", delta_cp)
                        print("cp_after_human_stm (opp POV at s'):", cp_after_human_stm)
                        print("cp_after_human (orig side POV)    :", cp_after_human)
                        print("human_delta (human - before)      :", human_delta)
                        print("Result-from-side-to-move z        :", z)
                        debug_plies_printed += 1

                # Advance along the *human* game line
                board.push(move)
                ply_idx += 1

            if game_count % 100 == 0:
                print(f"Processed {game_count} games, collected {len(X_list)} positions...")

    sf.close()

    if not X_list:
        raise RuntimeError("No positions collected; check PGN file and filters.")

    # Stack everything into arrays
    X = np.stack(X_list).astype(np.float32)
    y_policy_best = np.array(y_policy_best_list, dtype=np.int64)
    cp_before = np.array(cp_before_list, dtype=np.float32)
    cp_after_best = np.array(cp_after_best_list, dtype=np.float32)
    delta_cp = np.array(delta_cp_list, dtype=np.float32)
    human_move_idx = np.array(human_move_idx_list, dtype=np.int64)
    human_delta = np.array(human_delta_list, dtype=np.float32)
    game_result = np.array(game_result_list, dtype=np.float32)

    print(f"\nFinal dataset size: {X.shape[0]} positions")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUT_PATH,
        X=X,
        y_policy_best=y_policy_best,
        cp_before=cp_before,
        cp_after_best=cp_after_best,
        delta_cp=delta_cp,
        human_move_idx=human_move_idx,
        human_delta=human_delta,
        game_result=game_result,
    )

    print("Saved dataset to", OUT_PATH)
    print("X shape:", X.shape)
    print("y_policy_best shape:", y_policy_best.shape)
    print("cp_before range: ", cp_before.min(), cp_before.max())
    print("cp_before mean/std: ",
          float(cp_before.mean()), float(cp_before.std()))
    print("delta_cp range: ", delta_cp.min(), delta_cp.max())
    print("delta_cp mean/std: ",
          float(delta_cp.mean()), float(delta_cp.std()))
    print("human_move_idx valid fraction:",
          float(np.mean(human_move_idx >= 0)))
    print("human_delta mean/std: ",
          float(human_delta.mean()), float(human_delta.std()))
    print("game_result values:", np.unique(game_result))

    # -------- Post-hoc random sample debug ----------
    n = X.shape[0]
    if n > 0:
        print("\n=== Post-hoc random sample checks ===")
        num_samples = min(POSTHOC_DEBUG_SAMPLES, n)
        for _ in range(num_samples):
            i = random.randint(0, n - 1)
            stm_plane_val = X[i, 12, 0, 0]
            stm = "White" if stm_plane_val == 1.0 else "Black"
            print(f"\nSample {i}:")
            print(" side to move (from plane):", stm)
            print(" y_policy       :", y_policy_best[i])
            print(" cp_before            :", cp_before[i])
            print(" cp_after_best        :", cp_after_best[i])
            print(" recomputed delta_cp  :", cp_after_best[i] - cp_before[i])
            print(" stored delta_cp      :", delta_cp[i])
            print(" human_move_idx       :", human_move_idx[i])
            print(" human_delta          :", human_delta[i])
            print(" game_result (stm POV):", game_result[i])

        print("=== End post-hoc checks ===")


if __name__ == "__main__":
    main()
