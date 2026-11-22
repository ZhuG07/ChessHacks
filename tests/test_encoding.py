import chess
import numpy as np

from src.encoding import NUM_PLANES, POLICY_DIM, board_to_planes, index_to_move, move_to_index


def test_start_position_planes():
    planes = board_to_planes(chess.Board())
    assert planes.shape == (NUM_PLANES, 8, 8)
    assert planes.dtype == np.float32
    # 8 white pawns on rank 2 (row 6), 8 black pawns on rank 7 (row 1)
    assert planes[0, 6].sum() == 8
    assert planes[6, 1].sum() == 8
    # white king on e1 -> row 7, file 4
    assert planes[5, 7, 4] == 1
    # white to move, all castling rights, no en passant
    assert planes[12].all()
    assert all(planes[i].all() for i in range(13, 17))
    assert not planes[17].any()


def test_side_to_move_and_en_passant():
    board = chess.Board()
    board.push_san("e4")
    planes = board_to_planes(board)
    assert not planes[12].any()  # black to move
    # python-chess only reports an ep square when a capture is possible
    board.push_san("d5")
    board.push_san("e5")
    board.push_san("f5")
    planes = board_to_planes(board)
    assert planes[17, :, 5].all() and planes[17].sum() == 8


def test_move_index_roundtrip_for_every_legal_move():
    board = chess.Board("r3k2r/1P6/8/8/8/8/6p1/R3K2R w KQkq - 0 1")
    seen = set()
    for move in board.legal_moves:
        idx = move_to_index(move)
        assert 0 <= idx < POLICY_DIM
        assert index_to_move(idx) == move
        seen.add(idx)
    assert len(seen) == board.legal_moves.count()


def test_promotion_indices_are_distinct():
    a7a8 = [chess.Move.from_uci(f"a7a8{p}") for p in "qrbn"]
    indices = {move_to_index(m) for m in a7a8}
    indices.add(move_to_index(chess.Move.from_uci("a7a8")))
    assert len(indices) == 5
