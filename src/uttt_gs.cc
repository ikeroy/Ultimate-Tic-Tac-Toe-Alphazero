#include "uttt_gs.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace alphazero::uttt_gs {

namespace {

constexpr std::array<std::array<int, 3>, 8> kWinLines = {{
    {{0, 1, 2}}, {{3, 4, 5}}, {{6, 7, 8}}, {{0, 3, 6}},
    {{1, 4, 7}}, {{2, 5, 8}}, {{0, 4, 8}}, {{2, 4, 6}},
}};

}  // namespace

// Layout: 10*9 int8 board (90B) | int8 player (1B) | uint32 turn (4B) = 95B
static constexpr size_t kSerializedSize = 90 + 1 + 4;

UltimateTicTacToeGS::UltimateTicTacToeGS() {
  board_.setZero();
  board_(9, 0) = NO_MOVE;
  board_(9, 1) = NO_MOVE;
}

UltimateTicTacToeGS::UltimateTicTacToeGS(BoardTensor board, int8_t player,
                                         uint32_t turn)
    : board_(std::move(board)), player_(player), turn_(turn) {}

UltimateTicTacToeGS::UltimateTicTacToeGS(BoardTensor&& board, int8_t player,
                                         uint32_t turn)
    : board_(std::move(board)), player_(player), turn_(turn) {}

[[nodiscard]] std::unique_ptr<GameState> UltimateTicTacToeGS::copy() const
    noexcept {
  return std::make_unique<UltimateTicTacToeGS>(board_, player_, turn_);
}

[[nodiscard]] bool UltimateTicTacToeGS::operator==(
    const GameState& other) const noexcept {
  const auto* other_gs = dynamic_cast<const UltimateTicTacToeGS*>(&other);
  if (other_gs == nullptr) {
    return false;
  }
  for (int r = 0; r < BOARD_ROWS; ++r) {
    for (int c = 0; c < BOARD_COLS; ++c) {
      if (other_gs->board_(r, c) != board_(r, c)) {
        return false;
      }
    }
  }
  return other_gs->player_ == player_;
}

void UltimateTicTacToeGS::hash(absl::HashState h) const {
  h = absl::HashState::combine_contiguous(std::move(h), board_.data(),
                                          board_.size());
  absl::HashState::combine(std::move(h), player_);
}

void UltimateTicTacToeGS::group_cell_to_physical(int group, int cell, int* row,
                                                 int* col) noexcept {
  *row = (group / 3) * 3 + (cell / 3);
  *col = (group % 3) * 3 + (cell % 3);
}

void UltimateTicTacToeGS::physical_to_group_cell(int row, int col, int* group,
                                                 int* cell) noexcept {
  *group = (row / 3) * 3 + (col / 3);
  *cell = (row % 3) * 3 + (col % 3);
}

std::pair<int, int> UltimateTicTacToeGS::rotate_coord(int row, int col,
                                                      int turns) noexcept {
  turns = ((turns % 4) + 4) % 4;
  for (int i = 0; i < turns; ++i) {
    const int next_row = col;
    const int next_col = BOARD_SIDE - 1 - row;
    row = next_row;
    col = next_col;
  }
  return {row, col};
}

std::pair<int, int> UltimateTicTacToeGS::transform_coord(int row, int col,
                                                         int turns,
                                                         bool flip) noexcept {
  auto [r, c] = rotate_coord(row, col, turns);
  if (flip) {
    c = BOARD_SIDE - 1 - c;
  }
  return {r, c};
}

uint32_t UltimateTicTacToeGS::transform_action(uint32_t action, int turns,
                                               bool flip) noexcept {
  const int group = static_cast<int>(action / BOARD_CELLS);
  const int cell = static_cast<int>(action % BOARD_CELLS);
  int row = 0;
  int col = 0;
  group_cell_to_physical(group, cell, &row, &col);
  auto [new_row, new_col] = transform_coord(row, col, turns, flip);
  int new_group = 0;
  int new_cell = 0;
  physical_to_group_cell(new_row, new_col, &new_group, &new_cell);
  return static_cast<uint32_t>(new_group * BOARD_CELLS + new_cell);
}

int8_t UltimateTicTacToeGS::group_winner(uint8_t group) const noexcept {
  for (const auto& line : kWinLines) {
    const auto a = board_(group, line[0]);
    if (a == 0) {
      continue;
    }
    if (board_(group, line[1]) == a && board_(group, line[2]) == a) {
      return a;
    }
  }
  return 0;
}

int8_t UltimateTicTacToeGS::check_group_win(uint8_t group) const noexcept {
  return group_winner(group);
}

bool UltimateTicTacToeGS::group_full(uint8_t group) const noexcept {
  for (int c = 0; c < BOARD_CELLS; ++c) {
    if (board_(group, c) == 0) {
      return false;
    }
  }
  return true;
}

bool UltimateTicTacToeGS::group_playable(uint8_t group) const noexcept {
  return group_winner(group) == 0 && !group_full(group);
}

bool UltimateTicTacToeGS::move_wins_group(int group, int cell,
                                          int8_t piece) const noexcept {
  if (group < 0 || group >= BOARD_GROUPS || cell < 0 || cell >= BOARD_CELLS ||
      piece == 0 || !group_playable(static_cast<uint8_t>(group)) ||
      board_(group, cell) != 0) {
    return false;
  }
  for (const auto& line : kWinLines) {
    bool wins_line = true;
    bool contains_cell = false;
    for (const auto c : line) {
      contains_cell = contains_cell || c == cell;
      const auto value = c == cell ? piece : board_(group, c);
      if (value != piece) {
        wins_line = false;
        break;
      }
    }
    if (contains_cell && wins_line) {
      return true;
    }
  }
  return false;
}

bool UltimateTicTacToeGS::move_wins_board(int group, int cell,
                                          int8_t piece) const noexcept {
  if (!move_wins_group(group, cell, piece)) {
    return false;
  }
  std::array<int8_t, BOARD_GROUPS> wins{};
  for (int g = 0; g < BOARD_GROUPS; ++g) {
    wins[g] = group_winner(static_cast<uint8_t>(g));
  }
  wins[group] = piece;
  for (const auto& line : kWinLines) {
    const auto a = wins[line[0]];
    if (a == 0) {
      continue;
    }
    if (wins[line[1]] == a && wins[line[2]] == a) {
      return a == piece;
    }
  }
  return false;
}

bool UltimateTicTacToeGS::player_has_local_win_threat(
    int group, int8_t piece, int exclude_cell, int8_t exclude_piece) const
    noexcept {
  if (group < 0 || group >= BOARD_GROUPS || piece == 0 ||
      !group_playable(static_cast<uint8_t>(group))) {
    return false;
  }
  for (int cell = 0; cell < BOARD_CELLS; ++cell) {
    if (cell == exclude_cell || board_(group, cell) != 0) {
      continue;
    }
    for (const auto& line : kWinLines) {
      bool wins_line = true;
      bool contains_cell = false;
      for (const auto c : line) {
        contains_cell = contains_cell || c == cell;
        int8_t value = board_(group, c);
        if (c == cell) {
          value = piece;
        } else if (c == exclude_cell) {
          value = exclude_piece;
        }
        if (value != piece) {
          wins_line = false;
          break;
        }
      }
      if (contains_cell && wins_line) {
        return true;
      }
    }
  }
  return false;
}

bool UltimateTicTacToeGS::move_sends_to_local_win_threat(
    int group, int cell, int8_t piece, int8_t opponent_piece) const noexcept {
  if (group < 0 || group >= BOARD_GROUPS || cell < 0 || cell >= BOARD_CELLS ||
      board_(group, cell) != 0 || piece == 0 || opponent_piece == 0) {
    return false;
  }

  if (move_gives_opponent_free_choice(group, cell, piece)) {
    for (int g = 0; g < BOARD_GROUPS; ++g) {
      if (player_has_local_win_threat(g, opponent_piece,
                                      group == g ? cell : -1,
                                      group == g ? piece : 0)) {
        return true;
      }
    }
    return false;
  }

  const int destination = cell;
  if (!group_playable(static_cast<uint8_t>(destination))) {
    return false;
  }
  return player_has_local_win_threat(
      destination, opponent_piece, group == destination ? cell : -1,
      group == destination ? piece : 0);
}

bool UltimateTicTacToeGS::move_gives_opponent_free_choice(
    int group, int cell, int8_t piece) const noexcept {
  const auto destination = static_cast<uint8_t>(cell);
  if (!group_playable(destination)) {
    return true;
  }
  if (group != cell) {
    return false;
  }

  bool full_after_move = true;
  for (int c = 0; c < BOARD_CELLS; ++c) {
    if (c != cell && board_(group, c) == 0) {
      full_after_move = false;
      break;
    }
  }
  if (full_after_move) {
    return true;
  }

  for (const auto& line : kWinLines) {
    bool wins_line = true;
    for (const auto c : line) {
      const auto value = c == cell ? piece : board_(group, c);
      if (value != piece) {
        wins_line = false;
        break;
      }
    }
    if (wins_line) {
      return true;
    }
  }
  return false;
}

int8_t UltimateTicTacToeGS::board_winner_absolute() const noexcept {
  std::array<int8_t, BOARD_GROUPS> wins{};
  for (int g = 0; g < BOARD_GROUPS; ++g) {
    wins[g] = group_winner(static_cast<uint8_t>(g));
  }
  for (const auto& line : kWinLines) {
    const auto a = wins[line[0]];
    if (a == 0) {
      continue;
    }
    if (wins[line[1]] == a && wins[line[2]] == a) {
      return a;
    }
  }
  return 0;
}

int8_t UltimateTicTacToeGS::check_board_win(int8_t color) const noexcept {
  const auto win = board_winner_absolute();
  if (win != 0) {
    return static_cast<int8_t>(color * win);
  }
  return 0;
}

bool UltimateTicTacToeGS::has_legal_moves() const noexcept {
  return valid_moves().sum() > 0;
}

[[nodiscard]] Vector<uint8_t> UltimateTicTacToeGS::valid_moves() const noexcept {
  auto valids = Vector<uint8_t>{NUM_MOVES};
  valids.setZero();

  const int last_group = std::abs(static_cast<int>(board_(9, 0)));
  const int last_cell = std::abs(static_cast<int>(board_(9, 1)));
  const bool no_last_move = (last_group == NO_MOVE && last_cell == NO_MOVE);

  if (no_last_move) {
    for (int g = 0; g < BOARD_GROUPS; ++g) {
      for (int c = 0; c < BOARD_CELLS; ++c) {
        if (board_(g, c) != 0) {
          return valids;
        }
        valids(g * BOARD_CELLS + c) = 1;
      }
    }
    return valids;
  }

  if (last_cell < 0 || last_cell >= BOARD_GROUPS) {
    return valids;
  }

  if (!group_playable(static_cast<uint8_t>(last_cell))) {
    for (int g = 0; g < BOARD_GROUPS; ++g) {
      if (!group_playable(static_cast<uint8_t>(g))) {
        continue;
      }
      for (int c = 0; c < BOARD_CELLS; ++c) {
        if (board_(g, c) == 0) {
          valids(g * BOARD_CELLS + c) = 1;
        }
      }
    }
    return valids;
  }

  for (int c = 0; c < BOARD_CELLS; ++c) {
    if (board_(last_cell, c) == 0) {
      valids(last_cell * BOARD_CELLS + c) = 1;
    }
  }
  return valids;
}

void UltimateTicTacToeGS::play_move(uint32_t move) {
  if (move >= NUM_MOVES) {
    throw std::runtime_error{"Invalid move: You have a bug in your code."};
  }
  auto valids = valid_moves();
  if (valids(move) == 0) {
    throw std::runtime_error{"Invalid move: You have a bug in your code."};
  }

  const int group = static_cast<int>(move / BOARD_CELLS);
  const int cell = static_cast<int>(move % BOARD_CELLS);
  board_(group, cell) = player_sign();
  board_(9, 0) = static_cast<int8_t>(group);
  board_(9, 1) = static_cast<int8_t>(cell);
  player_ = static_cast<int8_t>(1 - player_);
  ++turn_;
}

[[nodiscard]] std::optional<Vector<float>> UltimateTicTacToeGS::scores()
    const noexcept {
  auto out = SizedVector<float, 3>{};
  out.setZero();
  const auto result = check_board_win(1);
  if (result == 0 && has_legal_moves()) {
    return std::nullopt;
  }
  if (result > 0) {
    out(0) = 1;
  } else if (result < 0) {
    out(1) = 1;
  } else {
    out(2) = 1;
  }
  return out;
}

[[nodiscard]] Tensor<float, 3> UltimateTicTacToeGS::canonicalized() const
    noexcept {
  auto out = CanonicalTensor{};
  out.setZero();
  const auto sign = player_sign();
  const auto valids = valid_moves();
  std::array<int8_t, BOARD_GROUPS> group_winners{};
  std::array<bool, BOARD_GROUPS> group_closed{};
  for (int g = 0; g < BOARD_GROUPS; ++g) {
    group_winners[g] = group_winner(static_cast<uint8_t>(g));
    group_closed[g] =
        group_winners[g] != 0 || group_full(static_cast<uint8_t>(g));
  }
  for (int g = 0; g < BOARD_GROUPS; ++g) {
    const auto winner = group_winners[g];
    const bool closed = group_closed[g];
    for (int c = 0; c < BOARD_CELLS; ++c) {
      int row = 0;
      int col = 0;
      group_cell_to_physical(g, c, &row, &col);
      const auto piece = board_(g, c);
      out(CURRENT_PLAYER_CELLS, row, col) = piece == sign ? 1.0f : 0.0f;
      out(OPPONENT_CELLS, row, col) = piece == -sign ? 1.0f : 0.0f;
      out(LEGAL_MOVES, row, col) = valids(g * BOARD_CELLS + c);
      out(CURRENT_PLAYER_WON_BOARDS, row, col) =
          winner == sign ? 1.0f : 0.0f;
      out(OPPONENT_WON_BOARDS, row, col) =
          winner == -sign ? 1.0f : 0.0f;
      out(CLOSED_BOARDS, row, col) = closed ? 1.0f : 0.0f;
      if (valids(g * BOARD_CELLS + c) != 0 &&
          (group_closed[c] ||
           (g == c && move_gives_opponent_free_choice(g, c, sign)))) {
        out(GIVES_OPPONENT_FREE_CHOICE, row, col) = 1.0f;
      }
      if (!closed && piece == 0) {
        if (move_wins_group(g, c, sign)) {
          out(CURRENT_LOCAL_WIN_THREATS, row, col) = 1.0f;
        }
        if (move_wins_group(g, c, static_cast<int8_t>(-sign))) {
          out(OPPONENT_LOCAL_WIN_THREATS, row, col) = 1.0f;
        }
        if (move_wins_board(g, c, sign)) {
          out(CURRENT_MACRO_WIN_THREATS, row, col) = 1.0f;
        }
        if (move_wins_board(g, c, static_cast<int8_t>(-sign))) {
          out(OPPONENT_MACRO_WIN_THREATS, row, col) = 1.0f;
        }
        if (valids(g * BOARD_CELLS + c) != 0 &&
            move_sends_to_local_win_threat(
                g, c, sign, static_cast<int8_t>(-sign))) {
          out(SENDS_OPPONENT_TO_LOCAL_WIN, row, col) = 1.0f;
        }
        if (move_sends_to_local_win_threat(
                g, c, static_cast<int8_t>(-sign), sign)) {
          out(OPPONENT_SENDS_CURRENT_TO_LOCAL_WIN, row, col) = 1.0f;
        }
      }
    }
  }

  const int last_group = std::abs(static_cast<int>(board_(9, 0)));
  const int last_cell = std::abs(static_cast<int>(board_(9, 1)));
  if (last_group != NO_MOVE || last_cell != NO_MOVE) {
    if (last_group >= 0 && last_group < BOARD_GROUPS && last_cell >= 0 &&
        last_cell < BOARD_GROUPS) {
      int row = 0;
      int col = 0;
      group_cell_to_physical(last_group, last_cell, &row, &col);
      out(LAST_MOVE, row, col) = 1.0f;
    }
  }

  if (last_cell >= 0 && last_cell < BOARD_GROUPS &&
      group_playable(static_cast<uint8_t>(last_cell))) {
    for (int c = 0; c < BOARD_CELLS; ++c) {
      int row = 0;
      int col = 0;
      group_cell_to_physical(last_cell, c, &row, &col);
      out(FORCED_BOARD, row, col) = 1.0f;
    }
  }
  return out;
}

[[nodiscard]] std::vector<PlayHistory> UltimateTicTacToeGS::symmetries(
    const PlayHistory& base) const noexcept {
  std::vector<PlayHistory> syms;
  syms.reserve(NUM_SYMMETRIES);
  for (int rot = 0; rot < 4; ++rot) {
    for (bool flip : {false, true}) {
      PlayHistory ph;
      ph.v = base.v;
      ph.canonical = CanonicalTensor{};
      ph.canonical.setZero();
      ph.pi = Vector<float>{NUM_MOVES};
      ph.pi.setZero();
      for (int p = 0; p < CANONICAL_SHAPE[0]; ++p) {
        for (int r = 0; r < BOARD_SIDE; ++r) {
          for (int c = 0; c < BOARD_SIDE; ++c) {
            const auto [nr, nc] = transform_coord(r, c, rot, flip);
            ph.canonical(p, nr, nc) = base.canonical(p, r, c);
          }
        }
      }
      for (uint32_t a = 0; a < NUM_MOVES; ++a) {
        ph.pi(transform_action(a, rot, flip)) = base.pi(a);
      }
      syms.push_back(std::move(ph));
    }
  }
  return syms;
}

[[nodiscard]] std::string UltimateTicTacToeGS::dump() const noexcept {
  std::string out = "Current Player: " + std::to_string(player_) + '\n';
  out += "Turn: " + std::to_string(turn_) + '\n';
  const auto separator = [](int row) {
    if (row == 0) {
      return "┌──────┬──────┬──────┐\n";
    }
    if (row == BOARD_SIDE) {
      return "└──────┴──────┴──────┘\n";
    }
    return "├──────┼──────┼──────┤\n";
  };
  for (int row = 0; row < BOARD_SIDE; ++row) {
    if (row % 3 == 0) {
      out += separator(row);
    }
    for (int col = 0; col < BOARD_SIDE; ++col) {
      if (col % 3 == 0) {
        out += "│";
      }
      int group = 0;
      int cell = 0;
      physical_to_group_cell(row, col, &group, &cell);
      auto piece = group_winner(static_cast<uint8_t>(group));
      if (piece == 0) {
        piece = board_(group, cell);
      }
      out += (piece == 1 ? 'X' : piece == -1 ? 'O' : '-');
      out += ' ';
    }
    out += "│\n";
  }
  out += separator(BOARD_SIDE);
  return out;
}

[[nodiscard]] std::string UltimateTicTacToeGS::to_bytes() const {
  std::string out(kSerializedSize, '\0');
  std::memcpy(&out[0], board_.data(), 90);
  out[90] = static_cast<char>(player_);
  std::memcpy(&out[91], &turn_, 4);
  return out;
}

[[nodiscard]] UltimateTicTacToeGS UltimateTicTacToeGS::from_bytes(
    const std::string& data) {
  if (data.size() != kSerializedSize) {
    throw std::runtime_error(
        "UltimateTicTacToeGS::from_bytes: wrong byte length");
  }
  BoardTensor board{};
  std::memcpy(board.data(), &data[0], 90);
  const auto player = static_cast<int8_t>(data[90]);
  uint32_t turn = 0;
  std::memcpy(&turn, &data[91], 4);
  return UltimateTicTacToeGS(board, player, turn);
}

}  // namespace alphazero::uttt_gs
