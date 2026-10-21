#include "uttt_gs.h"

#include "gtest/gtest.h"

namespace alphazero::uttt_gs {
namespace {

BoardTensor MakeEmptyBoard() {
  BoardTensor board{};
  board.setZero();
  board(9, 0) = NO_MOVE;
  board(9, 1) = NO_MOVE;
  return board;
}

void FillGroupWin(BoardTensor& board, int group, int8_t player) {
  board(group, 0) = player;
  board(group, 1) = player;
  board(group, 2) = player;
}

void FillGroupDraw(BoardTensor& board, int group) {
  const std::array<int8_t, BOARD_CELLS> cells = {1, -1, 1, 1, -1,
                                                 -1, -1, 1, -1};
  for (int c = 0; c < BOARD_CELLS; ++c) {
    board(group, c) = cells[c];
  }
}

float PlaneSum(const CanonicalTensor& tensor, int plane) {
  float sum = 0.0f;
  for (int r = 0; r < BOARD_SIDE; ++r) {
    for (int c = 0; c < BOARD_SIDE; ++c) {
      sum += tensor(plane, r, c);
    }
  }
  return sum;
}

void ExpectGroupPlane(const CanonicalTensor& tensor, int plane, int group,
                      float expected) {
  for (int cell = 0; cell < BOARD_CELLS; ++cell) {
    const int row = (group / 3) * 3 + (cell / 3);
    const int col = (group % 3) * 3 + (cell % 3);
    EXPECT_EQ(tensor(plane, row, col), expected)
        << "plane=" << plane << " group=" << group << " cell=" << cell;
  }
}

float AtMove(const CanonicalTensor& tensor, int plane, int move) {
  const int group = move / BOARD_CELLS;
  const int cell = move % BOARD_CELLS;
  const int row = (group / 3) * 3 + (cell / 3);
  const int col = (group % 3) * 3 + (cell % 3);
  return tensor(plane, row, col);
}

// NOLINTNEXTLINE
TEST(UltimateTicTacToeGS, DefaultState) {
  auto gs = UltimateTicTacToeGS{};
  EXPECT_EQ(gs.current_player(), 0);
  EXPECT_EQ(gs.current_turn(), 0U);
  EXPECT_EQ(gs.num_moves(), static_cast<uint32_t>(NUM_MOVES));
  EXPECT_EQ(gs.num_players(), 2);
  EXPECT_TRUE(gs.relative_values());
  EXPECT_TRUE(gs.has_legal_moves());
  EXPECT_EQ(gs.scores(), std::nullopt);

  auto valids = gs.valid_moves();
  auto expected = Vector<uint8_t>{NUM_MOVES};
  expected.setConstant(1);
  EXPECT_EQ(valids, expected);

  auto canonical = gs.canonicalized();
  EXPECT_EQ(canonical.dimension(0), CANONICAL_CHANNELS);
  EXPECT_EQ(canonical.dimension(1), BOARD_SIDE);
  EXPECT_EQ(canonical.dimension(2), BOARD_SIDE);
  EXPECT_EQ(PlaneSum(canonical, LEGAL_MOVES), 81.0f);
  for (const auto plane : {
           CURRENT_PLAYER_CELLS,
           OPPONENT_CELLS,
           LAST_MOVE,
           FORCED_BOARD,
           CURRENT_PLAYER_WON_BOARDS,
           OPPONENT_WON_BOARDS,
           CLOSED_BOARDS,
           GIVES_OPPONENT_FREE_CHOICE,
           CURRENT_LOCAL_WIN_THREATS,
           OPPONENT_LOCAL_WIN_THREATS,
           CURRENT_MACRO_WIN_THREATS,
           OPPONENT_MACRO_WIN_THREATS,
           SENDS_OPPONENT_TO_LOCAL_WIN,
           OPPONENT_SENDS_CURRENT_TO_LOCAL_WIN,
       }) {
    EXPECT_EQ(PlaneSum(canonical, plane), 0.0f) << "plane=" << plane;
  }
}

// NOLINTNEXTLINE
TEST(UltimateTicTacToeGS, PlayMoveAndCanonicalize) {
  auto gs = UltimateTicTacToeGS{};
  gs.play_move(0);

  EXPECT_EQ(gs.current_player(), 1);
  EXPECT_EQ(gs.current_turn(), 1U);

  auto valids = gs.valid_moves();
  EXPECT_EQ(valids.sum(), 8);
  EXPECT_EQ(valids(0), 0);
  for (int i = 1; i < 9; ++i) {
    EXPECT_EQ(valids(i), 1);
  }

  auto canonical = gs.canonicalized();
  EXPECT_EQ(canonical(CURRENT_PLAYER_CELLS, 0, 0), 0.0f);
  EXPECT_EQ(canonical(OPPONENT_CELLS, 0, 0), 1.0f);
  EXPECT_EQ(canonical(LAST_MOVE, 0, 0), 1.0f);
  EXPECT_EQ(PlaneSum(canonical, LEGAL_MOVES), 8.0f);
  EXPECT_EQ(PlaneSum(canonical, FORCED_BOARD), 9.0f);
  ExpectGroupPlane(canonical, FORCED_BOARD, 0, 1.0f);
  EXPECT_EQ(PlaneSum(canonical, GIVES_OPPONENT_FREE_CHOICE), 0.0f);

  gs.play_move(1);
  EXPECT_EQ(gs.current_player(), 0);
  EXPECT_EQ(gs.current_turn(), 2U);

  canonical = gs.canonicalized();
  EXPECT_EQ(canonical(CURRENT_PLAYER_CELLS, 0, 0), 1.0f);
  EXPECT_EQ(canonical(OPPONENT_CELLS, 0, 1), 1.0f);
  EXPECT_EQ(canonical(LAST_MOVE, 0, 1), 1.0f);
  EXPECT_EQ(PlaneSum(canonical, FORCED_BOARD), 9.0f);
  ExpectGroupPlane(canonical, FORCED_BOARD, 1, 1.0f);
}

// NOLINTNEXTLINE
TEST(UltimateTicTacToeGS, CapturedGroupAllowsAnywhere) {
  auto board = MakeEmptyBoard();
  FillGroupWin(board, 0, 1);
  board(9, 0) = 1;
  board(9, 1) = 0;
  auto gs = UltimateTicTacToeGS{board, 0, 4};

  auto valids = gs.valid_moves();
  EXPECT_EQ(valids.sum(), 72);
  EXPECT_EQ(valids(0), 0);
  EXPECT_EQ(valids(1), 0);
  EXPECT_EQ(valids(9), 1);
  EXPECT_EQ(valids(80), 1);

  const auto canonical = gs.canonicalized();
  EXPECT_EQ(PlaneSum(canonical, FORCED_BOARD), 0.0f);
  EXPECT_EQ(PlaneSum(canonical, LEGAL_MOVES), 72.0f);
  ExpectGroupPlane(canonical, CURRENT_PLAYER_WON_BOARDS, 0, 1.0f);
  ExpectGroupPlane(canonical, CLOSED_BOARDS, 0, 1.0f);
  EXPECT_EQ(PlaneSum(canonical, CURRENT_PLAYER_WON_BOARDS), 9.0f);
  EXPECT_EQ(PlaneSum(canonical, OPPONENT_WON_BOARDS), 0.0f);
  EXPECT_EQ(PlaneSum(canonical, CLOSED_BOARDS), 9.0f);
  EXPECT_EQ(PlaneSum(canonical, GIVES_OPPONENT_FREE_CHOICE), 8.0f);
  for (int group = 1; group < BOARD_GROUPS; ++group) {
    EXPECT_EQ(AtMove(canonical, GIVES_OPPONENT_FREE_CHOICE,
                     group * BOARD_CELLS),
              1.0f);
  }
}

// NOLINTNEXTLINE
TEST(UltimateTicTacToeGS, FullDrawnTargetGroupAllowsAnywhere) {
  auto board = MakeEmptyBoard();
  FillGroupDraw(board, 0);
  board(9, 0) = 1;
  board(9, 1) = 0;
  auto gs = UltimateTicTacToeGS{board, 0, 10};

  auto valids = gs.valid_moves();
  EXPECT_EQ(valids.sum(), 72);
  for (int c = 0; c < BOARD_CELLS; ++c) {
    EXPECT_EQ(valids(c), 0);
  }
  EXPECT_EQ(valids(9), 1);
  EXPECT_EQ(valids(80), 1);

  const auto canonical = gs.canonicalized();
  EXPECT_EQ(PlaneSum(canonical, FORCED_BOARD), 0.0f);
  ExpectGroupPlane(canonical, CLOSED_BOARDS, 0, 1.0f);
  EXPECT_EQ(PlaneSum(canonical, CURRENT_PLAYER_WON_BOARDS), 0.0f);
  EXPECT_EQ(PlaneSum(canonical, OPPONENT_WON_BOARDS), 0.0f);
  EXPECT_EQ(PlaneSum(canonical, GIVES_OPPONENT_FREE_CHOICE), 8.0f);
}

// NOLINTNEXTLINE
TEST(UltimateTicTacToeGS, WonBoardPlanesAreRelativeToCurrentPlayer) {
  auto board = MakeEmptyBoard();
  FillGroupWin(board, 0, 1);
  FillGroupWin(board, 1, -1);
  board(9, 0) = 2;
  board(9, 1) = 2;

  const auto player_zero =
      UltimateTicTacToeGS{board, 0, 6}.canonicalized();
  ExpectGroupPlane(player_zero, CURRENT_PLAYER_WON_BOARDS, 0, 1.0f);
  ExpectGroupPlane(player_zero, OPPONENT_WON_BOARDS, 1, 1.0f);
  ExpectGroupPlane(player_zero, CLOSED_BOARDS, 0, 1.0f);
  ExpectGroupPlane(player_zero, CLOSED_BOARDS, 1, 1.0f);
  EXPECT_EQ(PlaneSum(player_zero, CURRENT_PLAYER_CELLS), 3.0f);
  EXPECT_EQ(PlaneSum(player_zero, OPPONENT_CELLS), 3.0f);

  const auto player_one =
      UltimateTicTacToeGS{board, 1, 6}.canonicalized();
  ExpectGroupPlane(player_one, OPPONENT_WON_BOARDS, 0, 1.0f);
  ExpectGroupPlane(player_one, CURRENT_PLAYER_WON_BOARDS, 1, 1.0f);
  EXPECT_EQ(PlaneSum(player_one, CLOSED_BOARDS), 18.0f);
}

// NOLINTNEXTLINE
TEST(UltimateTicTacToeGS, ForcedBoardIncludesOccupiedCellsButLegalPlaneDoesNot) {
  auto board = MakeEmptyBoard();
  board(4, 0) = 1;
  board(4, 8) = -1;
  board(9, 0) = 2;
  board(9, 1) = 4;
  const auto canonical = UltimateTicTacToeGS{board, 0, 3}.canonicalized();

  ExpectGroupPlane(canonical, FORCED_BOARD, 4, 1.0f);
  EXPECT_EQ(PlaneSum(canonical, FORCED_BOARD), 9.0f);
  EXPECT_EQ(PlaneSum(canonical, LEGAL_MOVES), 7.0f);
  EXPECT_EQ(AtMove(canonical, LEGAL_MOVES, 4 * BOARD_CELLS), 0.0f);
  EXPECT_EQ(AtMove(canonical, LEGAL_MOVES, 4 * BOARD_CELLS + 8), 0.0f);
  EXPECT_EQ(AtMove(canonical, LEGAL_MOVES, 4 * BOARD_CELLS + 1), 1.0f);
}

// NOLINTNEXTLINE
TEST(UltimateTicTacToeGS, FreeChoiceTacticalPlaneOnlyMarksLegalMoves) {
  auto board = MakeEmptyBoard();
  FillGroupWin(board, 0, 1);
  board(9, 0) = 1;
  board(9, 1) = 0;
  const auto canonical = UltimateTicTacToeGS{board, 0, 4}.canonicalized();

  EXPECT_EQ(AtMove(canonical, GIVES_OPPONENT_FREE_CHOICE, 0), 0.0f);
  EXPECT_EQ(AtMove(canonical, GIVES_OPPONENT_FREE_CHOICE, 1), 0.0f);
  EXPECT_EQ(AtMove(canonical, GIVES_OPPONENT_FREE_CHOICE, 9), 1.0f);
  EXPECT_EQ(AtMove(canonical, GIVES_OPPONENT_FREE_CHOICE, 10), 0.0f);
}

// NOLINTNEXTLINE
TEST(UltimateTicTacToeGS, MoveThatWinsItsOwnDestinationBoardGrantsFreeChoice) {
  auto board = MakeEmptyBoard();
  board(4, 3) = 1;
  board(4, 5) = 1;
  board(9, 0) = 0;
  board(9, 1) = 4;
  const auto canonical = UltimateTicTacToeGS{board, 0, 3}.canonicalized();

  const int winning_move = 4 * BOARD_CELLS + 4;
  ASSERT_EQ(AtMove(canonical, LEGAL_MOVES, winning_move), 1.0f);
  EXPECT_EQ(AtMove(canonical, GIVES_OPPONENT_FREE_CHOICE, winning_move),
            1.0f);
  EXPECT_EQ(AtMove(canonical, GIVES_OPPONENT_FREE_CHOICE,
                   4 * BOARD_CELLS + 2),
            0.0f);
}

// NOLINTNEXTLINE
TEST(UltimateTicTacToeGS, MoveThatFillsItsOwnDestinationBoardGrantsFreeChoice) {
  auto board = MakeEmptyBoard();
  FillGroupDraw(board, 4);
  board(4, 4) = 0;
  board(9, 0) = 0;
  board(9, 1) = 4;
  const auto canonical = UltimateTicTacToeGS{board, 0, 8}.canonicalized();

  const int filling_move = 4 * BOARD_CELLS + 4;
  ASSERT_EQ(AtMove(canonical, LEGAL_MOVES, filling_move), 1.0f);
  EXPECT_EQ(AtMove(canonical, GIVES_OPPONENT_FREE_CHOICE, filling_move),
            1.0f);
}

// NOLINTNEXTLINE
TEST(UltimateTicTacToeGS, LocalThreatPlanesMarkCurrentAndOpponentThreats) {
  auto board = MakeEmptyBoard();
  board(4, 3) = 1;
  board(4, 5) = 1;
  board(5, 0) = -1;
  board(5, 1) = -1;
  board(9, 0) = 0;
  board(9, 1) = 4;
  const auto canonical = UltimateTicTacToeGS{board, 0, 6}.canonicalized();

  EXPECT_EQ(AtMove(canonical, CURRENT_LOCAL_WIN_THREATS,
                   4 * BOARD_CELLS + 4),
            1.0f);
  EXPECT_EQ(AtMove(canonical, OPPONENT_LOCAL_WIN_THREATS,
                   5 * BOARD_CELLS + 2),
            1.0f);
  EXPECT_EQ(AtMove(canonical, LEGAL_MOVES, 5 * BOARD_CELLS + 2), 0.0f);
}

// NOLINTNEXTLINE
TEST(UltimateTicTacToeGS, MacroThreatPlanesMarkBoardWinningCaptures) {
  auto board = MakeEmptyBoard();
  FillGroupWin(board, 0, 1);
  FillGroupWin(board, 1, 1);
  board(2, 0) = 1;
  board(2, 1) = 1;
  FillGroupWin(board, 3, -1);
  FillGroupWin(board, 4, -1);
  board(5, 3) = -1;
  board(5, 4) = -1;
  board(9, 0) = 8;
  board(9, 1) = 8;
  const auto canonical = UltimateTicTacToeGS{board, 0, 10}.canonicalized();

  EXPECT_EQ(AtMove(canonical, CURRENT_MACRO_WIN_THREATS,
                   2 * BOARD_CELLS + 2),
            1.0f);
  EXPECT_EQ(AtMove(canonical, OPPONENT_MACRO_WIN_THREATS,
                   5 * BOARD_CELLS + 5),
            1.0f);
}

// NOLINTNEXTLINE
TEST(UltimateTicTacToeGS, DestinationDangerPlaneMarksLegalBadSends) {
  auto board = MakeEmptyBoard();
  board(4, 0) = -1;
  board(4, 1) = -1;
  board(9, 0) = 0;
  board(9, 1) = 3;
  const auto canonical = UltimateTicTacToeGS{board, 0, 4}.canonicalized();

  EXPECT_EQ(AtMove(canonical, SENDS_OPPONENT_TO_LOCAL_WIN,
                   3 * BOARD_CELLS + 4),
            1.0f);
  EXPECT_EQ(AtMove(canonical, SENDS_OPPONENT_TO_LOCAL_WIN,
                   3 * BOARD_CELLS + 5),
            0.0f);
}

// NOLINTNEXTLINE
TEST(UltimateTicTacToeGS, OpponentDestinationDangerPlaneMarksPossibleBadSends) {
  auto board = MakeEmptyBoard();
  board(7, 3) = 1;
  board(7, 4) = 1;
  board(9, 0) = 0;
  board(9, 1) = 3;
  const auto canonical = UltimateTicTacToeGS{board, 0, 4}.canonicalized();

  EXPECT_EQ(AtMove(canonical, OPPONENT_SENDS_CURRENT_TO_LOCAL_WIN,
                   2 * BOARD_CELLS + 7),
            1.0f);
}

// NOLINTNEXTLINE
TEST(UltimateTicTacToeGS, FullBoardWithoutMacroWinIsDraw) {
  auto board = MakeEmptyBoard();
  for (int g = 0; g < BOARD_GROUPS; ++g) {
    FillGroupDraw(board, g);
  }
  board(9, 0) = 8;
  board(9, 1) = 8;
  auto gs = UltimateTicTacToeGS{board, 0, 81};

  EXPECT_FALSE(gs.has_legal_moves());
  EXPECT_EQ(gs.check_board_win(1), 0);
  auto scores = gs.scores();
  ASSERT_TRUE(scores.has_value());
  EXPECT_EQ((*scores)(0), 0.0f);
  EXPECT_EQ((*scores)(1), 0.0f);
  EXPECT_EQ((*scores)(2), 1.0f);

  const auto canonical = gs.canonicalized();
  EXPECT_EQ(PlaneSum(canonical, CLOSED_BOARDS), 81.0f);
  EXPECT_EQ(PlaneSum(canonical, LEGAL_MOVES), 0.0f);
  EXPECT_EQ(PlaneSum(canonical, FORCED_BOARD), 0.0f);
  EXPECT_EQ(PlaneSum(canonical, GIVES_OPPONENT_FREE_CHOICE), 0.0f);
}

// NOLINTNEXTLINE
TEST(UltimateTicTacToeGS, InvalidInitialStateHasNoMoves) {
  auto board = MakeEmptyBoard();
  board(0, 0) = 1;
  auto gs = UltimateTicTacToeGS{board, 0, 0};

  auto valids = gs.valid_moves();
  EXPECT_EQ(valids.sum(), 0);
  EXPECT_FALSE(gs.has_legal_moves());
  const auto canonical = gs.canonicalized();
  EXPECT_EQ(PlaneSum(canonical, LEGAL_MOVES), 0.0f);
  EXPECT_EQ(PlaneSum(canonical, GIVES_OPPONENT_FREE_CHOICE), 0.0f);
}

// NOLINTNEXTLINE
TEST(UltimateTicTacToeGS, BoardWinAndScores) {
  auto board = MakeEmptyBoard();
  FillGroupWin(board, 0, 1);
  FillGroupWin(board, 1, 1);
  FillGroupWin(board, 2, 1);
  auto gs = UltimateTicTacToeGS{board, 0, 3};

  EXPECT_EQ(gs.check_group_win(0), 1);
  EXPECT_EQ(gs.check_board_win(1), 1);
  EXPECT_EQ(gs.check_board_win(-1), -1);

  auto scores = gs.scores();
  ASSERT_TRUE(scores.has_value());
  EXPECT_EQ((*scores)(0), 1.0f);
  EXPECT_EQ((*scores)(1), 0.0f);
  EXPECT_EQ((*scores)(2), 0.0f);
}

// NOLINTNEXTLINE
TEST(UltimateTicTacToeGS, CopyAndBytesRoundTrip) {
  auto gs = UltimateTicTacToeGS{};
  gs.play_move(0);
  gs.play_move(1);
  auto copy = gs.copy();
  EXPECT_EQ(gs, *copy);

  auto bytes = gs.to_bytes();
  auto restored = UltimateTicTacToeGS::from_bytes(bytes);
  EXPECT_EQ(gs, restored);
  EXPECT_EQ(restored.current_player(), gs.current_player());
  EXPECT_EQ(restored.current_turn(), gs.current_turn());
}

// NOLINTNEXTLINE
TEST(UltimateTicTacToeGS, SymmetriesTransformBoardAndPolicy) {
  PlayHistory base;
  base.canonical = CanonicalTensor{};
  base.canonical.setZero();
  for (int plane = 0; plane < CANONICAL_CHANNELS; ++plane) {
    base.canonical(plane, plane / BOARD_SIDE, plane % BOARD_SIDE) = 1.0f;
  }
  base.pi = Vector<float>{NUM_MOVES};
  base.pi.setZero();
  base.pi(0) = 1.0f;
  base.v = Vector<float>{3};
  base.v.setZero();
  base.v(2) = 1.0f;

  auto syms = UltimateTicTacToeGS{}.symmetries(base);
  EXPECT_EQ(syms.size(), 8U);

  bool found_rot90 = false;
  for (const auto& ph : syms) {
    for (int plane = 0; plane < CANONICAL_CHANNELS; ++plane) {
      EXPECT_EQ(PlaneSum(ph.canonical, plane), 1.0f) << "plane=" << plane;
    }
    EXPECT_EQ(ph.pi.sum(), 1.0f);
    EXPECT_EQ(ph.v(2), 1.0f);
    if (ph.canonical(CURRENT_PLAYER_CELLS, 0, 8) == 1.0f &&
        ph.pi(20) == 1.0f) {
      found_rot90 = true;
    }
  }
  EXPECT_TRUE(found_rot90);
}

}  // namespace
}  // namespace alphazero::uttt_gs
