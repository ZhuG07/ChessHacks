import chess
import pytest

from src.search import MATE_SCORE, mate_distance, win_probability


def test_mate_distance():
    assert mate_distance(MATE_SCORE - 1) == 1
    assert mate_distance(MATE_SCORE - 3) == 2
    assert mate_distance(-(MATE_SCORE - 2)) == -1
    assert mate_distance(250) is None


def test_win_probability_is_monotonic_and_bounded():
    assert win_probability(0) == pytest.approx(0.5)
    assert win_probability(300) > win_probability(100) > 0.5
    assert win_probability(MATE_SCORE) == 1.0
    assert win_probability(-MATE_SCORE) == 0.0


@pytest.mark.parametrize(
    "fen, expected",
    [
        ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1a8"),  # back-rank mate
        ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "f3f7"),  # scholar's mate
        ("6k1/8/6K1/8/8/8/8/R7 w - - 0 1", "a1a8"),
    ],
)
def test_finds_mate_in_one(engine, fen, expected):
    result = engine.analyse(chess.Board(fen), time_limit=1.0)
    assert result.best_move.uci() == expected
    assert result.mate_in == 1


def test_wins_a_hanging_queen(engine):
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p1q1/3P4/2N5/PPP1PPPP/R1BQKBNR w KQkq - 0 1")
    result = engine.analyse(board, time_limit=1.0)
    assert result.best_move.uci() == "c1g5"


def test_result_is_legal_and_consistent(engine):
    board = chess.Board()
    for san in ["e4", "e5", "Nf3", "Nc6", "Bb5"]:
        board.push_san(san)
    result = engine.analyse(board, time_limit=0.5)
    assert result.best_move in board.legal_moves
    assert result.depth >= 1
    assert result.pv and result.pv[0] == result.best_move
    # the principal variation must be playable from the root
    b = board.copy()
    for mv in result.pv:
        assert mv in b.legal_moves
        b.push(mv)
    # MultiPV lines are sorted best first and cover every root move
    exact = [l.score for l in result.lines if l.exact]
    assert exact == sorted(exact, reverse=True)
    assert {l.move for l in result.lines} == set(board.legal_moves)
    assert sum(result.policy.values()) == pytest.approx(1.0, abs=1e-5)


def test_search_does_not_mutate_board(engine):
    board = chess.Board()
    board.push_san("d4")
    before = (board.fen(), list(board.move_stack))
    engine.analyse(board, time_limit=0.3)
    assert (board.fen(), list(board.move_stack)) == before


def test_no_legal_moves(engine):
    mated = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    result = engine.analyse(mated, time_limit=0.2)
    assert result.best_move is None
    assert result.score == -MATE_SCORE


def test_instinct_plays_top_policy_move(engine):
    board = chess.Board()
    result = engine.instinct(board)
    assert result.depth == 0
    assert result.best_move == max(result.policy, key=result.policy.get)


def test_material_balance_is_from_side_to_move():
    from src.search import material_balance

    assert material_balance(chess.Board()) == 0
    up_a_queen = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert material_balance(up_a_queen) == 900
    up_a_queen.turn = chess.BLACK
    assert material_balance(up_a_queen) == -900


def test_evaluation_knows_a_queen_is_a_lot(engine):
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")
    result = engine.analyse(board, time_limit=0.3)
    assert result.static_eval > 500


def test_does_not_walk_into_a_discovered_defence(engine):
    # From a benchmark game: 12...Bg5 attacks the queen but loses a piece to
    # 13.Qxg5, because the c1 bishop then defends g5. The first version of
    # the search played it.
    board = chess.Board("r1b1k2r/ppppbppp/8/4q3/1n2P3/2N2B2/PPPQ1PPP/R1BR2K1 b kq - 14 12")
    result = engine.analyse(board, time_limit=1.0)
    assert board.san(result.best_move) != "Bg5"
