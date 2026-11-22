"""
Board and move encoding shared by training, self-play and inference.

Input planes (18 x 8 x 8, float32):
    0-5    white pawn, knight, bishop, rook, queen, king
    6-11   black pawn, knight, bishop, rook, queen, king
    12     side to move (all ones if white to move)
    13-16  castling rights: white K, white Q, black K, black Q
    17     en-passant file (the whole file is set)

Row 0 of each plane is rank 8, so the tensor reads like a diagram from
white's side.

Policy index:
    (from_square * 64 + to_square) * 5 + promo_idx
with promo_idx in [none, queen, rook, bishop, knight]. That gives
64 * 64 * 5 = 20480 logits, most of which are never legal; the engine only
ever looks at the logits of legal moves.
"""

from __future__ import annotations

import chess
import numpy as np

NUM_PLANES = 18
PROMO_PIECES = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
NUM_PROMOS = len(PROMO_PIECES)
POLICY_DIM = 64 * 64 * NUM_PROMOS

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

_PROMO_INDEX = {piece: i for i, piece in enumerate(PROMO_PIECES)}


def board_to_planes(board: chess.Board) -> np.ndarray:
    """Encode a position as an (18, 8, 8) float32 array."""
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for sq, piece in board.piece_map().items():
        idx = PIECE_PLANES[(piece.piece_type, piece.color)]
        planes[idx, 7 - chess.square_rank(sq), chess.square_file(sq)] = 1.0

    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16, :, :] = 1.0

    if board.ep_square is not None:
        planes[17, :, chess.square_file(board.ep_square)] = 1.0

    return planes


def move_to_index(move: chess.Move) -> int:
    """Map a move to its slot in the 20480-way policy output."""
    promo_idx = _PROMO_INDEX[move.promotion]
    return (move.from_square * 64 + move.to_square) * NUM_PROMOS + promo_idx


def index_to_move(index: int) -> chess.Move:
    """Inverse of move_to_index (the move is not checked for legality)."""
    square_pair, promo_idx = divmod(index, NUM_PROMOS)
    from_sq, to_sq = divmod(square_pair, 64)
    return chess.Move(from_sq, to_sq, promotion=PROMO_PIECES[promo_idx])
