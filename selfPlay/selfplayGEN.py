# training/selfplay/selfplay_generator.py

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import chess
import torch  # OK to keep; used by the engine, and harmless here

# How often to deliberately play a bad move (to learn to handle blunders)
BLUNDER_PLAY_PROB = 0.25         # 25% of the time we play a "bad" move
BLUNDER_CP_DROP_THRESHOLD = 150  # centipawns worse than best to qualify as a blunder

from .config import (
    SELFPLAY_OUT_DIR,
    MODEL_PATH,
    TOP_K_MOVES,
    MAX_MOVES_PER_GAME,
    NUM_GAMES,
    SF_TIME_LIMIT,
    POLICY_TEMP_CP,
    CP_SCALE,
    SELFPLAY_ENGINE_DEPTH,
)

from .stockfishWRAP import StockfishEvaluator

# Engine / model side
from src.main import (
    PolicyOnlyEngine,
    board_to_planes,
    move_to_index,
    POLICY_DIM,
)


def softmax_from_cp(cp_values: np.ndarray, temp_cp: float) -> np.ndarray:
    """
    Convert centipawn evals to a probability distribution using a temperature
    measured in centipawns.
    """
    if cp_values.size == 0:
        return cp_values

    scaled = cp_values.astype(np.float32) / float(temp_cp)
    max_s = scaled.max()
    exps = np.exp(scaled - max_s)
    Z = exps.sum()
    if Z <= 0:
        return np.ones_like(exps) / len(exps)
    return exps / Z


def _interesting_weight(
    board: chess.Board,
    best_sf_move: chess.Move,
    cp_before: float,
    cp_after_best: float,
) -> int:
    """
    Simple reward shaping via sample weighting:
      - midgame / endgame emphasized
      - bonuses for good checks and mating lines
    Returns an integer >= 1 used to duplicate the sample.
    """

    weight = 1

    # Game phase: emphasize midgame & endgame
    fullmove = board.fullmove_number
    if fullmove >= 12:
        weight += 1        # midgame
    if fullmove >= 25:
        weight += 1        # deeper endgame

    # Check / mate bonuses
    # Check if best SF move gives check
    board.push(best_sf_move)
    is_check = board.is_check()
    board.pop()

    cp_gain = cp_after_best - cp_before

    # Good checking move (check + eval improves a bit)
    if is_check and cp_gain > 30:
        weight += 1

    # Near-mate or mate (our wrapper uses ±100000 for mates)
    if abs(cp_after_best) >= 90_000:
        weight += 3

    # Cap the weight to avoid insane duplication
    return max(1, min(weight, 5))


def _would_cause_threefold(board: chess.Board, move: chess.Move) -> bool:
    """
    Check if playing `move` would allow a threefold repetition claim.
    """
    board.push(move)
    can = board.can_claim_threefold_repetition()
    board.pop()
    return can


def play_one_selfplay_game(
    engine: PolicyOnlyEngine,
    sf: StockfishEvaluator,
    top_k: int,
    max_moves: int,
) -> Tuple[
    list[np.ndarray],
    list[int],
    list[np.ndarray],
    list[float],
    list[float],
    list[float],
    list[int],
]:
    """
    Play a single self-play game with your NN+alpha-beta engine as the actor
    and Stockfish as the critic on top-K root moves.

    CHECKMATE-ONLY:
      - If the game does not end in checkmate, this function returns
        empty lists, so no positions from that game are added to the dataset.

    Returns lists over plies:
      - planes_list:      list of (18, 8, 8) float32
      - policy_best_idx:  list of ints (Stockfish-best among top-K)
      - policy_topk_idx:  list of np.array(K,) of ints
      - cp_before_list:   list of floats
      - cp_after_best:    list of floats
      - result_list:      list of floats (game outcome from stm POV)
      - weight_list:      list of ints (sample duplication weight)
    """

    board = chess.Board()

    planes_list: list[np.ndarray] = []
    policy_best_idx_list: list[int] = []
    policy_topk_idx_list: list[np.ndarray] = []
    cp_before_list: list[float] = []
    cp_after_best_list: list[float] = []
    weight_list: list[int] = []

    move_count = 0

    # for computing result from White POV:
    board_history: list[chess.Board] = []

    while not board.is_game_over() and move_count < max_moves:
        board_history.append(board.copy(stack=False))

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break

        # Ask your engine for move + root probs (search-based)
        # Use a shallower depth for self-play than for tournaments for speed.
        best_move, probs_dict = engine.search_best_move(
            board,
            max_depth=SELFPLAY_ENGINE_DEPTH,
        )
        if best_move is None:
            # Safety fallback: random legal move
            best_move = random.choice(legal_moves)

        # Ensure probs over legal moves
        probs = np.array(
            [probs_dict.get(mv, 0.0) for mv in legal_moves],
            dtype=np.float32,
        )
        Z = probs.sum()
        if Z <= 0:
            probs[:] = 1.0 / len(legal_moves)
        else:
            probs /= Z

        # Choose top-K moves by engine's probability (actor's proposal set)
        sorted_indices = np.argsort(-probs)
        k = min(top_k, len(legal_moves))
        topk_indices = sorted_indices[:k]
        topk_moves = [legal_moves[i] for i in topk_indices]

        # Stockfish evaluates current position and each top-K move
        cp_before = sf.evaluate_cp(board)

        cp_after = []
        for mv in topk_moves:
            board.push(mv)
            cp_after_m = sf.evaluate_cp(board)
            board.pop()
            cp_after.append(cp_after_m)
        cp_after = np.array(cp_after, dtype=np.float32)

        # choose SF-best among top-K
        best_sf_idx_local = int(cp_after.argmax())
        best_sf_move = topk_moves[best_sf_idx_local]
        best_sf_cp = float(cp_after[best_sf_idx_local])

        # build sparse policy target info (indices into global POLICY_DIM space)
        policy_topk_idx = np.array(
            [move_to_index(mv) for mv in topk_moves],
            dtype=np.int64,
        )

        # (Optional) soft distribution over top-K from cp_after
        _policy_topk_prob = softmax_from_cp(cp_after, temp_cp=POLICY_TEMP_CP)
        # NOTE: we don't store it yet; you can extend NPZ later to use it.

        # record features & labels (except game result)
        planes = board_to_planes(board)
        planes_list.append(planes)
        policy_best_idx_list.append(move_to_index(best_sf_move))
        policy_topk_idx_list.append(policy_topk_idx)
        cp_before_list.append(float(cp_before))
        cp_after_best_list.append(best_sf_cp)

        # reward-shaping weight for this sample
        w = _interesting_weight(board, best_sf_move, cp_before, best_sf_cp)
        weight_list.append(w)

        # ------------------------------------------------------------------
        # Decide which move to actually PLAY in self-play:
        #
        # Goals:
        #   - Mostly play strong moves (so games are sane).
        #   - Sometimes deliberately play a *bad* move so the engine learns:
        #       * how to punish blunders
        #       * how to defend after its own blunders
        #   - When Stockfish says "mate", ALWAYS follow the mating line.
        #   - Avoid threefold repetition when possible.
        # ------------------------------------------------------------------

        r = random.random()
        chosen: chess.Move

        # 1) If SF says this move is essentially mate (or near-mate),
        #    ALWAYS follow best_sf_move to ensure clean mating sequences.
        if abs(best_sf_cp) >= 90_000:
            chosen = best_sf_move

        else:
            # 2) Otherwise, we mix blunders and normal exploration.

            if r < BLUNDER_PLAY_PROB and len(topk_moves) > 1:
                # --- BLUNDER MODE ---
                # Find moves that are clearly worse than the best SF move
                best_cp = float(cp_after[best_sf_idx_local])
                cp_drop = best_cp - cp_after  # positive if worse than best

                # Indices where move is "bad enough"
                blunder_indices = np.where(cp_drop >= BLUNDER_CP_DROP_THRESHOLD)[0]

                if blunder_indices.size == 0:
                    # If no clearly bad moves in top-K, just take the worst among them
                    worst_idx_local = int(cp_after.argmin())
                    chosen = topk_moves[worst_idx_local]
                else:
                    # Randomly pick one of the clearly bad moves
                    idx_local = int(np.random.choice(blunder_indices))
                    chosen = topk_moves[idx_local]

            else:
                # --- NORMAL MODE ---
                # Opening: more exploration for uncommon lines
                if board.fullmove_number <= 12:
                    explore_prob = 0.4  # 60% best, 40% explore
                else:
                    explore_prob = 0.2  # 80% best, 20% explore

                if random.random() < (1.0 - explore_prob):
                    # Play engine/search best move (which is often SF-best too)
                    chosen = best_move
                else:
                    # Sample from NN's move distribution for diversity
                    chosen = random.choices(
                        legal_moves,
                        weights=probs.tolist(),
                        k=1
                    )[0]

        # 3) Avoid causing threefold repetition if we can
        if _would_cause_threefold(board, chosen):
            alt_moves = [m for m in legal_moves if m != chosen and not _would_cause_threefold(board, m)]
            if alt_moves:
                chosen = random.choice(alt_moves)

        board.push(chosen)
        move_count += 1

    # ------------------------------------------------------------------
    # CHECKMATE-ONLY FILTER
    # ------------------------------------------------------------------

    # If the game ended by move limit and is not technically game-over,
    # or it is over but not by checkmate (stalemate, repetition, etc.),
    # we discard this game entirely.
    if not board.is_game_over() or not board.is_checkmate():
        # No usable data from this game
        print("[SELFPLAY] Game ended without checkmate; discarding positions from this game.")
        return [], [], [], [], [], [], []

    # At the end of the game, we compute result per position.
    result_str = board.result()  # e.g. "1-0", "0-1", "1/2-1/2"
    if result_str == "1-0":
        z_white = 1.0
    elif result_str == "0-1":
        z_white = -1.0
    else:
        z_white = 0.0

    # For each stored position, result from side-to-move POV
    result_list: list[float] = []
    for b in board_history[: len(planes_list)]:  # ensure same length
        stm_factor = 1.0 if b.turn == chess.WHITE else -1.0
        result_list.append(z_white * stm_factor)

    return (
        planes_list,
        policy_best_idx_list,
        policy_topk_idx_list,
        cp_before_list,
        cp_after_best_list,
        result_list,
        weight_list,
    )


def generate_selfplay_npz(
    out_path: Path,
    num_games: int = NUM_GAMES,
    top_k: int = TOP_K_MOVES,
    max_moves: int = MAX_MOVES_PER_GAME,
    sf_time_limit: float = SF_TIME_LIMIT,
):
    """
    Generate a self-play + Stockfish-labeled dataset and save it to `out_path`.
    The network used is loaded via PolicyOnlyEngine / MODEL_PATH.

    Only positions from games that end in CHECKMATE are included.
    """
    print(f"[SELFPLAY] Loading NN engine from {MODEL_PATH}")
    engine = PolicyOnlyEngine()  # uses your existing CONFIG + MODEL_PATH in src/main.py
    engine.reset()
    sf = StockfishEvaluator(time_limit=sf_time_limit)

    X_all: list[np.ndarray] = []
    policy_best_idx_all: list[int] = []
    policy_topk_idx_all: list[np.ndarray] = []
    cp_before_all: list[float] = []
    cp_after_best_all: list[float] = []
    delta_cp_all: list[float] = []
    result_all: list[float] = []

    try:
        for gi in range(num_games):
            print(f"[SELFPLAY] Game {gi + 1}/{num_games}")
            (
                planes_list,
                policy_best_idx_list,
                policy_topk_idx_list,
                cp_before_list,
                cp_after_best_list,
                result_list,
                weight_list,
            ) = play_one_selfplay_game(
                engine=engine,
                sf=sf,
                top_k=top_k,
                max_moves=max_moves,
            )

            # If this game was discarded (non-checkmate), skip
            if not planes_list:
                continue

            for i in range(len(planes_list)):
                # Duplicate positions according to their weight to emphasize
                # good checks / mates / mid/endgame.
                repeat = max(1, int(weight_list[i]))
                for _ in range(repeat):
                    X_all.append(planes_list[i])
                    policy_best_idx_all.append(policy_best_idx_list[i])
                    policy_topk_idx_all.append(policy_topk_idx_list[i])
                    cp_before_all.append(cp_before_list[i])
                    cp_after_best_all.append(cp_after_best_list[i])
                    delta_cp_all.append(cp_after_best_list[i] - cp_before_list[i])
                    result_all.append(result_list[i])

        if not X_all:
            print("[SELFPLAY] No positions generated; aborting.")
            return

        # Stack X
        X_arr = np.stack(X_all).astype(np.float32)
        policy_best_idx_arr = np.array(policy_best_idx_all, dtype=np.int64)

        # ------------------------------------------------------------------
        # policy_topk_idx_all is a list of arrays of length k, where
        #   k = min(TOP_K_MOVES, len(legal_moves)) at that position.
        # In endgames or forced positions, len(legal_moves) < TOP_K_MOVES,
        # so the arrays have different lengths and np.stack() fails.
        #
        # Fix: pad each array with -1 up to max_k, then stack.
        # (-1 means "no move here" — you can safely ignore or mask later.)
        # ------------------------------------------------------------------
        max_k = max(arr.shape[0] for arr in policy_topk_idx_all)

        policy_topk_idx_padded = []
        for arr in policy_topk_idx_all:
            if arr.shape[0] < max_k:
                pad_len = max_k - arr.shape[0]
                pad = -np.ones(pad_len, dtype=np.int64)
                arr = np.concatenate([arr, pad])
            policy_topk_idx_padded.append(arr)

        policy_topk_idx_arr = np.stack(policy_topk_idx_padded).astype(np.int64)

        cp_before_arr = np.array(cp_before_all, dtype=np.float32)
        cp_after_best_arr = np.array(cp_after_best_all, dtype=np.float32)
        delta_cp_arr = np.array(delta_cp_all, dtype=np.float32)
        result_arr = np.array(result_all, dtype=np.float32)

        print(f"[SELFPLAY] Saving {X_arr.shape[0]} positions to {out_path}")
        np.savez_compressed(
            out_path,
            X=X_arr,
            policy_best_idx=policy_best_idx_arr,
            policy_topk_idx=policy_topk_idx_arr,
            cp_before=cp_before_arr,
            cp_after_best=cp_after_best_arr,
            delta_cp=delta_cp_arr,
            result=result_arr,
        )

    finally:
        sf.close()
        engine.reset()


if __name__ == "__main__":
    out_file = SELFPLAY_OUT_DIR / "selfplay_sf_topk_dataset.npz"
    generate_selfplay_npz(out_file)
