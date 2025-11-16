# training/selfplay/stockfish_wrapper.py

from pathlib import Path
import chess
import chess.engine

from .config import STOCKFISH_PATH

class StockfishEvaluator:
    """
    Lightweight Stockfish wrapper used as a critic
    for self-play labeling.
    """

    def __init__(self, engine_path: str | Path = STOCKFISH_PATH, time_limit: float = 0.05, depth: int | None = None):
        self.engine_path = str(engine_path)
        self.time_limit = float(time_limit)
        self.depth = depth

        self.engine = chess.engine.SimpleEngine.popen_uci(str(self.engine_path))

    def evaluate_cp(self, board: chess.Board) -> int:
        """
        Evaluate the position from the perspective of the side to move.
        Returns centipawns (positive = good for side to move).
        """
        if self.depth is not None:
            limit = chess.engine.Limit(depth=self.depth)
        else:
            limit = chess.engine.Limit(time=self.time_limit)

        info = self.engine.analyse(board, limit=limit)

        score = info["score"].pov(board.turn)
        if score.is_mate():
            # map mate distance to large cp (sign only matters)
            sign = 1 if score.mate() > 0 else -1
            return 100000 * sign
        else:
            return score.score(mate_score=100000)

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass

    def __del__(self):
        self.close()
