"""
High-level engine object: load a checkpoint once, then ask for moves.

    from src.engine import Engine
    engine = Engine()                       # default checkpoint
    result = engine.analyse(board, time_limit=1.0)
    result.best_move, result.score, result.pv
"""

from __future__ import annotations

import os
from pathlib import Path

import chess
import torch

from .model import load_model
from .search import NeuralEvaluator, SearchConfig, SearchResult, Searcher

MODEL_DIR = Path(__file__).resolve().parent / "model_save"

# The checkpoint the engine plays with unless told otherwise. Can be
# overridden with the CHESS_ENGINE_CHECKPOINT environment variable (either a
# file name inside src/model_save or a full path).
DEFAULT_CHECKPOINT = "best_selfplay.pt"


def resolve_checkpoint(name: str | os.PathLike | None = None) -> Path:
    name = name or os.getenv("CHESS_ENGINE_CHECKPOINT") or DEFAULT_CHECKPOINT
    path = Path(name)
    if not path.is_absolute() and not path.exists():
        path = MODEL_DIR / path
    return path


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Engine:
    def __init__(self, checkpoint: str | os.PathLike | None = None,
                 device: str | torch.device | None = None,
                 config: SearchConfig | None = None):
        self.device = torch.device(device) if device else default_device()
        self.checkpoint = resolve_checkpoint(checkpoint)
        self.model, self.info = load_model(self.checkpoint, self.device)
        self.evaluator = NeuralEvaluator(self.model, self.device)
        self.searcher = Searcher(self.evaluator, config or SearchConfig())

    @property
    def config(self) -> SearchConfig:
        return self.searcher.cfg

    def analyse(self, board: chess.Board, time_limit: float | None = None,
                max_depth: int | None = None) -> SearchResult:
        return self.searcher.search(board, time_limit=time_limit, max_depth=max_depth)

    def instinct(self, board: chess.Board) -> SearchResult:
        return self.searcher.instinct(board)

    def best_move(self, board: chess.Board, time_limit: float | None = None) -> chess.Move | None:
        return self.analyse(board, time_limit=time_limit).best_move

    def new_game(self):
        self.searcher.reset()

    def describe(self) -> dict:
        return {
            **self.info,
            "device": str(self.device),
        }
