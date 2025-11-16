# training/selfplay/rl_loop.py

from __future__ import annotations

import argparse

import chess
import chess.pgn
import torch

from .config import (
    SELFPLAY_OUT_DIR,
    MODEL_PATH,          # supervised best.pt
    NUM_GAMES,
    TOP_K_MOVES,
    MAX_MOVES_PER_GAME,
    SF_TIME_LIMIT,
    SELFPLAY_ENGINE_DEPTH,
)

# RL-specific model path
RL_MODEL_PATH = MODEL_PATH.parent / "best_selfplay.pt"

# Your existing generator / trainer
from .selfplayGEN import generate_selfplay_npz
from .trainselfplay import train_selfplay_model

# Your engine (NN + alpha-beta search)
from src.main import PolicyOnlyEngine


def _build_rl_engine() -> PolicyOnlyEngine:
    """
    Build a PolicyOnlyEngine and, if RL weights exist, load them on top.
    This way, self-play uses the evolving RL model, but src/main.py can
    still use MODEL_PATH (best.pt) for other things.
    """
    engine = PolicyOnlyEngine()

    # Try to override weights with RL model if present
    if RL_MODEL_PATH.is_file():
        try:
            state = torch.load(RL_MODEL_PATH, map_location="cpu")
            engine.model.load_state_dict(state, strict=False)
            print(f"[RL_ENGINE] Loaded RL weights from {RL_MODEL_PATH}")
        except Exception as e:
            print(f"[RL_ENGINE] Failed to load RL weights ({e}), using base weights.")

    engine.reset()
    return engine


def run_one_cycle(
    cycle_idx: int,
    num_games: int,
    top_k: int = TOP_K_MOVES,
    max_moves: int = MAX_MOVES_PER_GAME,
    sf_time_limit: float = SF_TIME_LIMIT,
):
    """
    One RL iteration:
      1) Generate self-play data with current RL weights
      2) Fine-tune RL model on that data, saving back into RL_MODEL_PATH
    """
    # Overwrite a single dataset file each cycle (no infinite growth)
    dataset_path = SELFPLAY_OUT_DIR / "selfplay_sf_topk_dataset.npz"

    print("=" * 80)
    print(f"[CYCLE {cycle_idx}] Generating self-play dataset → {dataset_path}")

    # Inside generate_selfplay_npz, PolicyOnlyEngine will be instantiated.
    # To ensure it uses RL weights, you can either:
    #  - Keep using PolicyOnlyEngine in selfplayGEN as-is (it will load base MODEL_PATH),
    #  - OR modify selfplayGEN to call _build_rl_engine() instead.
    #
    # EASIEST: modify selfplayGEN to use RL weights too. See note below.
    generate_selfplay_npz(
        out_path=dataset_path,
        num_games=num_games,
        top_k=top_k,
        max_moves=max_moves,
        sf_time_limit=sf_time_limit,
    )

    print(f"[CYCLE {cycle_idx}] Training on self-play data (fine-tune RL model)...")

    # Now train RL-specific weights (best_selfplay.pt), recursively
    train_selfplay_model(dataset_path, out_model_path=RL_MODEL_PATH)

    print(f"[CYCLE {cycle_idx}] Done training, RL weights updated at {RL_MODEL_PATH}")


def visualize_self_play(max_ply: int = 120):
    """
    Visualize a single self-play game with the CURRENT RL weights (if available).

    This does NOT use Stockfish. It just lets your engine play itself and prints
    a PGN so you can see the style: openings, sacs, checks, etc.
    """
    print("-" * 80)
    print("[VISUALIZE] Playing a self-play game with current RL model...")

    engine = _build_rl_engine()

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = "RL-Engine"
    game.headers["Black"] = "RL-Engine"
    node = game

    ply = 0
    while not board.is_game_over() and ply < max_ply:
        best_move, _ = engine.search_best_move(board, max_depth=SELFPLAY_ENGINE_DEPTH)
        if best_move is None:
            print("[VISUALIZE] No legal move returned; stopping.")
            break

        board.push(best_move)
        node = node.add_variation(best_move)
        ply += 1

    if board.is_game_over():
        game.headers["Result"] = board.result()
    else:
        game.headers["Result"] = "*"

    print("[VISUALIZE] PGN of self-play game:")
    print()
    print(game)
    print()


def dump_checkmate_pgn(cycle_idx: int, max_tries: int = 5, max_ply: int = 200):
    """
    Try up to max_tries times to generate a self-play game that ends in checkmate
    using the CURRENT RL model, and dump it as a PGN file for inspection.
    """
    for attempt in range(1, max_tries + 1):
        print(f"[CYCLE {cycle_idx}] Generating PGN example game (attempt {attempt}/{max_tries})")

        engine = _build_rl_engine()

        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["White"] = "RL-Engine"
        game.headers["Black"] = "RL-Engine"

        node = game
        ply = 0

        while not board.is_game_over() and ply < max_ply:
            best_move, _ = engine.search_best_move(board, max_depth=SELFPLAY_ENGINE_DEPTH)
            if best_move is None:
                break

            board.push(best_move)
            node = node.add_variation(best_move)
            ply += 1

        if board.is_checkmate():
            game.headers["Result"] = board.result()
            pgn_str = str(game)

            pgn_path = SELFPLAY_OUT_DIR / f"cycle_{cycle_idx:03d}_example.pgn"
            with open(pgn_path, "w", encoding="utf-8") as f:
                f.write(pgn_str)

            print(f"[CYCLE {cycle_idx}] Saved checkmate PGN to {pgn_path}")
            print()
            print(pgn_str)
            print()
            return

        else:
            print(
                f"[CYCLE {cycle_idx}] Game did not end in checkmate "
                f"(result={board.result() if board.is_game_over() else 'incomplete'})."
            )

    print(f"[CYCLE {cycle_idx}] Failed to generate a checkmate game in {max_tries} attempts.")


def main():
    parser = argparse.ArgumentParser(
        description="Run RL self-play cycles (generate + train + visualize)."
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of (generate → train → visualize) RL cycles to run.",
    )
    parser.add_argument(
        "--games-per-cycle",
        type=int,
        default=NUM_GAMES,
        help="Number of self-play games per generation run.",
    )
    parser.add_argument(
        "--visualize-max-ply",
        type=int,
        default=120,
        help="Max number of plies to play in the visualization game.",
    )

    args = parser.parse_args()

    for c in range(args.cycles):
        run_one_cycle(
            cycle_idx=c,
            num_games=args.games_per_cycle,
            top_k=TOP_K_MOVES,
            max_moves=MAX_MOVES_PER_GAME,
            sf_time_limit=SF_TIME_LIMIT,
        )

        # Quick “live” visualization (could be draw or mate)
        visualize_self_play(max_ply=args.visualize_max_ply)

        # Guaranteed checkmate example (or best-effort)
        dump_checkmate_pgn(cycle_idx=c, max_tries=5, max_ply=200)

    print("=" * 80)
    print(
        f"RL loop finished. Latest RL model weights are in: {RL_MODEL_PATH}\n"
        f"Supervised best.pt remains at: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
