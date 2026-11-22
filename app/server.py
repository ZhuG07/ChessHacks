"""
Web server for playing against the engine in a browser.

    uvicorn app.server:app --port 8000

The server is stateless: every request carries the full list of moves played
so far (and optionally a starting FEN), and the board is rebuilt from that.
This keeps repetition detection correct and means any number of people can
play at once without the server tracking sessions.

Endpoints
    GET  /api/info       engine and network description
    POST /api/position   legal moves and game status for a position
    POST /api/move       let the engine pick and play a move
    POST /api/analyse    search a position without playing (hints, review)
    GET  /api/docs       interactive OpenAPI docs
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import chess
import chess.pgn
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.engine import Engine
from src.search import SearchResult, mate_distance, win_probability

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_THINK_SECONDS = float(os.getenv("MAX_THINK_SECONDS", "5"))

if os.getenv("TORCH_THREADS"):
    torch.set_num_threads(int(os.environ["TORCH_THREADS"]))

_engine: Engine | None = None
_engine_lock = threading.Lock()


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the weights and run one short search so the first real request
    # does not pay for model loading and warm-up.
    engine = get_engine()
    with _engine_lock:
        engine.analyse(chess.Board(), time_limit=0.2)
    yield


app = FastAPI(title="ChessHacks Engine", docs_url="/api/docs", redoc_url=None, lifespan=lifespan)


# ============================================================
# Request / response helpers
# ============================================================

class PositionRequest(BaseModel):
    moves: list[str] = Field(default_factory=list, max_length=1000, description="UCI moves from the start position")
    start_fen: str | None = Field(None, description="Optional starting FEN (defaults to the standard start)")


class SearchRequest(PositionRequest):
    think_time: float = Field(1.0, ge=0.05, description="Seconds the engine may think")
    max_depth: int | None = Field(None, ge=1, le=20)
    instinct: bool = Field(False, description="Play the network's top policy move with no search")


def build_board(req: PositionRequest) -> chess.Board:
    try:
        board = chess.Board(req.start_fen) if req.start_fen else chess.Board()
    except ValueError:
        raise HTTPException(400, "Invalid FEN")
    for uci in req.moves:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            raise HTTPException(400, f"Malformed move: {uci}")
        if move not in board.legal_moves:
            raise HTTPException(400, f"Illegal move: {uci}")
        board.push(move)
    return board


def game_status(board: chess.Board) -> dict:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return {"over": False, "result": None, "reason": None}
    reason = outcome.termination.name.replace("_", " ").lower()
    return {"over": True, "result": outcome.result(), "reason": reason}


def position_payload(board: chess.Board, start_fen: str | None) -> dict:
    replay = chess.Board(start_fen) if start_fen else chess.Board()
    san_history = []
    for mv in board.move_stack:
        san_history.append(replay.san(mv))
        replay.push(mv)

    legal = [
        {"uci": mv.uci(), "san": board.san(mv), "from": chess.square_name(mv.from_square),
         "to": chess.square_name(mv.to_square), "promotion": bool(mv.promotion)}
        for mv in board.legal_moves
    ]

    king = board.king(board.turn)
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "Casual game vs ChessHacks Engine"

    return {
        "fen": board.fen(),
        "turn": "w" if board.turn == chess.WHITE else "b",
        "legal": legal,
        "history": san_history,
        "check": chess.square_name(king) if board.is_check() and king is not None else None,
        "status": game_status(board),
        "pgn": str(game),
    }


def _white_pov(score: float, board: chess.Board) -> float:
    return score if board.turn == chess.WHITE else -score


def analysis_payload(result: SearchResult, board: chess.Board) -> dict:
    def score_info(score: float) -> dict:
        white = _white_pov(score, board)
        mate = mate_distance(white)
        return {"cp": None if mate is not None else round(white, 1), "mate": mate,
                "win": round(win_probability(white), 4)}

    lines = []
    for line in result.lines[:5]:
        lines.append({
            "move": line.move.uci(),
            "san": board.san(line.move),
            "pv": board.variation_san(line.pv) if line.pv else "",
            "pv_uci": [m.uci() for m in line.pv],
            "prior": round(line.prior, 4),
            "exact": line.exact,
            **score_info(line.score),
        })

    policy = sorted(result.policy.items(), key=lambda kv: -kv[1])[:8]
    win_from_result_head = result.win_prob if board.turn == chess.WHITE else 1 - result.win_prob

    return {
        "best_move": result.best_move.uci() if result.best_move else None,
        "best_san": board.san(result.best_move) if result.best_move else None,
        "score": score_info(result.score),
        "depth": result.depth,
        "nodes": result.nodes,
        "nn_evals": result.nn_evals,
        "time_ms": round(result.time_ms, 1),
        "nps": int(result.nodes / max(result.time_ms / 1000, 1e-6)),
        "pv": board.variation_san(result.pv) if result.pv else "",
        "lines": lines,
        "policy": [{"move": m.uci(), "san": board.san(m), "prob": round(p, 4)} for m, p in policy],
        "static_eval": round(_white_pov(result.static_eval, board), 1),
        "result_head_win": round(win_from_result_head, 4),
    }


def run_search(req: SearchRequest, board: chess.Board) -> SearchResult:
    engine = get_engine()
    with _engine_lock:
        if req.instinct:
            return engine.instinct(board)
        think = min(req.think_time, MAX_THINK_SECONDS)
        return engine.analyse(board, time_limit=think, max_depth=req.max_depth)


# ============================================================
# Routes
# ============================================================

@app.get("/api/info")
def info():
    engine = get_engine()
    cfg = engine.config
    return {
        "name": "ChessHacks Engine",
        "network": engine.describe(),
        "search": {
            "algorithm": "negamax alpha-beta, iterative deepening, MultiPV",
            "multipv": cfg.multipv,
            "prune_width": cfg.prune_width,
            "batch_size": cfg.batch_size,
            "max_think_seconds": MAX_THINK_SECONDS,
        },
        "torch_threads": torch.get_num_threads(),
    }


@app.post("/api/position")
def position(req: PositionRequest):
    board = build_board(req)
    return position_payload(board, req.start_fen)


@app.post("/api/move")
def move(req: SearchRequest):
    board = build_board(req)
    if board.is_game_over(claim_draw=True):
        raise HTTPException(409, "Game is already over")

    result = run_search(req, board)
    analysis = analysis_payload(result, board)
    played = result.best_move
    analysis["played_san"] = board.san(played)

    board.push(played)
    return {"move": played.uci(), "analysis": analysis, "position": position_payload(board, req.start_fen)}


@app.post("/api/analyse")
def analyse(req: SearchRequest):
    board = build_board(req)
    if board.is_game_over(claim_draw=True):
        return {"analysis": None, "position": position_payload(board, req.start_fen)}
    result = run_search(req, board)
    return {"analysis": analysis_payload(result, board), "position": position_payload(board, req.start_fen)}


@app.get("/healthz")
def health():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
