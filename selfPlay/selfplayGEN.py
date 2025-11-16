# training/selfplay/selfplay_generator.py

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import chess

import torch

from .config import (
    SELFPLAY_OUT_DIR,
    MODEL_PATH,
    TOP_K_MOVES,
    MAX_MOVES_PER_GAME,
    NUM_GAMES,
    SF_TIME_LIMIT,
    POLICY_TEMP_CP,
    CP_SCALE,
)
from .stockfish_wrapper import StockfishEvaluator

# Adjust this import to match your project layout:
# If engine.py sits under something like `submission/engine.py`, use that path.
from engine import (
    NNEvalEngine,
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


def play_one_selfplay_game(
    engine: NNEvalEngine,
    sf: StockfishEvaluator,
    top_k: int,
    max_moves: int,
) -> Tuple[list, list, list, list, list]:
    """
    Play a single self-play game with your NN+alpha-beta engine as the actor
    and Stockfish as the critic on top-K root moves.

    Returns lists over plies:
      - planes_list:     list of (18, 8, 8) float32
      - policy_best_idx: list of ints (Stockfish-best among top-K)
      - policy_topk_idx: list of np.array(K,) of ints
      - cp_before_list:  list of floats
      - cp_after_best:   list of floats
    Result labels (game outcome) are computed later after game end.
    """

    board = chess.Board()

    planes_list: list[np.ndarray] = []
    policy_best_idx_list: list[int] = []
    policy_topk_idx_list: list[np.ndarray] = []
    cp_before_list: list[float] = []
    cp_after_best_list: list[float] = []

    move_count = 0

    # for computing result from White POV:
    board_history: list[chess.Board] = []

    while not board.is_game_over() and move_count < max_moves:
        board_history.append(board.copy(stack=False))

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break

        # ask your engine for move + root probs
        engine.note_position(board)
        best_move, probs_dict = engine.select_move(board)
        if best_move is None:
            best_move = random.choice(legal_moves)

        # ensure probs over legal moves
        probs = np.array([probs_dict.get(mv, 0.0) for mv in legal_moves], dtype=np.float32)
        Z = probs.sum()
        if Z <= 0:
            probs[:] = 1.0 / len(legal_moves)
        else:
            probs /= Z

        # choose top-K moves by engine's probability (actor's proposal set)
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

        # build sparse policy target info
        policy_topk_idx = np.array(
            [move_to_index(mv) for mv in topk_moves],
            dtype=np.int64,
        )

        # we compute a soft distribution over top-K from cp_after
        policy_topk_prob = softmax_from_cp(cp_after, temp_cp=POLICY_TEMP_CP)

        # record features & labels (except game result)
        planes = board_to_planes(board)
        planes_list.append(planes)
        policy_best_idx_list.append(move_to_index(best_sf_move))
        policy_topk_idx_list.append(policy_topk_idx)
        cp_before_list.append(float(cp_before))
        cp_after_best_list.append(best_sf_cp)

        # decide which move to actually PLAY in self-play:
        # small exploration: 80% best_move, 20% sample from probs
        if random.random() < 0.8:
            chosen = best_move
        else:
            chosen = random.choices(legal_moves, weights=probs.tolist(), k=1)[0]

        board.push(chosen)
        move_count += 1

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
    The network used is loaded via NNEvalEngine / MODEL_PATH.
    """
    print(f"[SELFPLAY] Loading NN engine from {MODEL_PATH}")
    engine = NNEvalEngine()  # uses your existing CONFIG + MODEL_PATH
    sf = StockfishEvaluator(time_limit=sf_time_limit)

    X_all: list[np.ndarray] = []
    policy_best_idx_all: list[int] = []
    policy_topk_idx_all: list[np.ndarray] = []
    policy_topk_prob_all: list[np.ndarray] = []
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
            ) = play_one_selfplay_game(
                engine=engine,
                sf=sf,
                top_k=top_k,
                max_moves=max_moves,
            )

            for i in range(len(planes_list)):
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

        X_arr = np.stack(X_all).astype(np.float32)
        policy_best_idx_arr = np.array(policy_best_idx_all, dtype=np.int64)
        policy_topk_idx_arr = np.stack(policy_topk_idx_all).astype(np.int64)

        # For now we recompute policy_topk_prob from cp_after within each game;
        # alternatively, store them during play_one_selfplay_game.
        # Simpler version: uniform over top-K SF-labeled moves.
        # But better: use cp_after; let's rebuild:
        # For memory reasons, we’ll store only indices and let training
        # reconstruct per-position probabilities if desired.

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
