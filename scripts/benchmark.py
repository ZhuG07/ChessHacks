#!/usr/bin/env python3
"""
Play the engine against Stockfish with limited strength and report a score.

    python scripts/benchmark.py --elo 1320 1600 --games 10 --think 1.0

Stockfish's UCI_LimitStrength / UCI_Elo option is used as a reference
opponent. Games start from a fixed set of balanced openings and colours
alternate, so each opening is played once from each side. A game that
reaches --max-plies is scored as a draw.

Requires a Stockfish binary on PATH (or pass --stockfish).
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.engine import Engine  # noqa: E402

OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6",
    "d2d4 d7d5 c2c4 e7e6",
    "e2e4 c7c5 g1f3 d7d6",
    "d2d4 g8f6 c2c4 g7g6",
    "c2c4 e7e5 b1c3 g8f6",
    "e2e4 e7e6 d2d4 d7d5",
    "e2e4 c7c6 d2d4 d7d5",
    "g1f3 d7d5 g2g3 g8f6",
    "e2e4 e7e5 f1c4 g8f6",
    "d2d4 d7d5 c1f4 g8f6",
]


def elo_diff(score: float) -> float | None:
    """Elo difference implied by an expected score (None at 0% or 100%)."""
    if score <= 0 or score >= 1:
        return None
    return -400 * math.log10(1 / score - 1)


def play_game(engine: Engine, sf: chess.engine.SimpleEngine, opening: str, engine_white: bool,
              think: float, sf_time: float, max_plies: int) -> chess.pgn.Game:
    board = chess.Board()
    for uci in opening.split():
        board.push_uci(uci)
    engine.new_game()

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        if (board.turn == chess.WHITE) == engine_white:
            move = engine.best_move(board, time_limit=think)
        else:
            move = sf.play(board, chess.engine.Limit(time=sf_time)).move
        board.push(move)

    game = chess.pgn.Game.from_board(board)
    game.headers["White"] = "ChessHacks Engine" if engine_white else "Stockfish"
    game.headers["Black"] = "Stockfish" if engine_white else "ChessHacks Engine"
    result = board.result(claim_draw=True)
    game.headers["Result"] = "1/2-1/2" if result == "*" else result
    return game


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--elo", type=int, nargs="+", default=[1320], help="Stockfish UCI_Elo levels to play")
    parser.add_argument("--games", type=int, default=10, help="games per Elo level (even number recommended)")
    parser.add_argument("--think", type=float, default=1.0, help="engine seconds per move")
    parser.add_argument("--sf-time", type=float, default=0.1, help="Stockfish seconds per move")
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--checkpoint", default=None, help="checkpoint file (defaults to the engine default)")
    parser.add_argument("--stockfish", default=shutil.which("stockfish") or "/usr/games/stockfish")
    parser.add_argument("--threads", type=int, default=2, help="torch CPU threads")
    parser.add_argument("--pgn", type=Path, default=None, help="write all games to this PGN file")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    engine = Engine(args.checkpoint, device="cpu")
    print(f"Engine: {engine.describe()}")

    pgn_out = open(args.pgn, "w") if args.pgn else None
    summary = []

    for elo in args.elo:
        sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        sf.configure({"UCI_LimitStrength": True, "UCI_Elo": elo, "Threads": 1})
        wins = draws = losses = 0
        t0 = time.time()

        for g in range(args.games):
            opening = OPENINGS[(g // 2) % len(OPENINGS)]
            engine_white = g % 2 == 0
            game = play_game(engine, sf, opening, engine_white, args.think, args.sf_time, args.max_plies)
            result = game.headers["Result"]
            if result == "1/2-1/2":
                draws += 1
                outcome = "draw"
            elif (result == "1-0") == engine_white:
                wins += 1
                outcome = "win"
            else:
                losses += 1
                outcome = "loss"
            plies = len(list(game.mainline_moves()))
            print(f"  [Elo {elo}] game {g + 1}/{args.games}: engine {'white' if engine_white else 'black'}, "
                  f"{outcome} ({result}, {plies} plies)", flush=True)
            if pgn_out:
                game.headers["Event"] = f"Benchmark vs Stockfish UCI_Elo {elo}"
                print(game, file=pgn_out, end="\n\n", flush=True)

        sf.quit()
        n = wins + draws + losses
        score = (wins + 0.5 * draws) / n
        diff = elo_diff(score)
        diff_text = f"{diff:+.0f}" if diff is not None else "n/a"
        line = (f"Stockfish UCI_Elo {elo}: +{wins} ={draws} -{losses}  score {score:.0%}  "
                f"(implied Elo difference {diff_text})  [{time.time() - t0:.0f}s]")
        summary.append(line)
        print(line, flush=True)

    print("\nSummary")
    for line in summary:
        print(" ", line)
    if pgn_out:
        pgn_out.close()


if __name__ == "__main__":
    main()
