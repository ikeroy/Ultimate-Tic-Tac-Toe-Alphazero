#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dll_export.h"
#include "game_state.h"

namespace alphazero::uttt_gs {

constexpr int BOARD_GROUPS = 9;
constexpr int BOARD_CELLS = 9;
constexpr int BOARD_ROWS = 10;
constexpr int BOARD_COLS = 9;
constexpr int BOARD_SIDE = 9;
constexpr int NUM_MOVES = BOARD_GROUPS * BOARD_CELLS;
constexpr int NUM_PLAYERS = 2;
constexpr int NUM_SYMMETRIES = 8;
constexpr std::array<int, 2> BOARD_SHAPE = {BOARD_ROWS, BOARD_COLS};
constexpr int CURRENT_PLAYER_CELLS = 0;
constexpr int OPPONENT_CELLS = 1;
constexpr int LAST_MOVE = 2;
constexpr int LEGAL_MOVES = 3;
constexpr int FORCED_BOARD = 4;
constexpr int CURRENT_PLAYER_WON_BOARDS = 5;
constexpr int OPPONENT_WON_BOARDS = 6;
constexpr int CLOSED_BOARDS = 7;
constexpr int GIVES_OPPONENT_FREE_CHOICE = 8;
constexpr int CURRENT_LOCAL_WIN_THREATS = 9;
constexpr int OPPONENT_LOCAL_WIN_THREATS = 10;
constexpr int CURRENT_MACRO_WIN_THREATS = 11;
constexpr int OPPONENT_MACRO_WIN_THREATS = 12;
constexpr int SENDS_OPPONENT_TO_LOCAL_WIN = 13;
constexpr int OPPONENT_SENDS_CURRENT_TO_LOCAL_WIN = 14;
constexpr int CANONICAL_CHANNELS = 15;
constexpr std::array<int, 3> CANONICAL_SHAPE = {
    CANONICAL_CHANNELS, BOARD_SIDE, BOARD_SIDE};
constexpr int NO_MOVE = 10;

using BoardTensor =
    SizedTensor<int8_t, Eigen::Sizes<BOARD_ROWS, BOARD_COLS>>;
using CanonicalTensor =
    SizedTensor<float, Eigen::Sizes<CANONICAL_SHAPE[0], CANONICAL_SHAPE[1],
                                    CANONICAL_SHAPE[2]>>;

class DLLEXPORT UltimateTicTacToeGS : public GameState {
 public:
  UltimateTicTacToeGS();
  UltimateTicTacToeGS(BoardTensor board, int8_t player, uint32_t turn);
  UltimateTicTacToeGS(BoardTensor&& board, int8_t player, uint32_t turn);

  [[nodiscard]] std::unique_ptr<GameState> copy() const noexcept override;
  [[nodiscard]] bool operator==(const GameState& other) const noexcept override;
  void hash(absl::HashState h) const override;

  [[nodiscard]] uint8_t current_player() const noexcept override {
    return player_;
  }
  [[nodiscard]] uint32_t current_turn() const noexcept override { return turn_; }
  [[nodiscard]] uint32_t num_moves() const noexcept override { return NUM_MOVES; }
  [[nodiscard]] uint8_t num_players() const noexcept override { return 2; }
  [[nodiscard]] bool relative_values() const noexcept override { return true; }

  [[nodiscard]] Vector<uint8_t> valid_moves() const noexcept override;
  void play_move(uint32_t move) override;
  [[nodiscard]] std::optional<Vector<float>> scores() const noexcept override;
  [[nodiscard]] Tensor<float, 3> canonicalized() const noexcept override;
  [[nodiscard]] uint8_t num_symmetries() const noexcept override {
    return NUM_SYMMETRIES;
  }
  [[nodiscard]] std::vector<PlayHistory> symmetries(
      const PlayHistory& base) const noexcept override;
  [[nodiscard]] std::string dump() const noexcept override;

  [[nodiscard]] std::string to_bytes() const override;
  [[nodiscard]] static UltimateTicTacToeGS from_bytes(
      const std::string& data);

  [[nodiscard]] int8_t check_group_win(uint8_t group) const noexcept;
  [[nodiscard]] int8_t check_board_win(int8_t color = 1) const noexcept;
  [[nodiscard]] bool has_legal_moves() const noexcept;

 private:
  [[nodiscard]] int8_t player_sign() const noexcept {
    return player_ == 0 ? 1 : -1;
  }
  [[nodiscard]] int8_t group_winner(uint8_t group) const noexcept;
  [[nodiscard]] bool group_full(uint8_t group) const noexcept;
  [[nodiscard]] bool group_playable(uint8_t group) const noexcept;
  [[nodiscard]] int8_t board_winner_absolute() const noexcept;
  [[nodiscard]] bool move_wins_group(int group, int cell,
                                     int8_t piece) const noexcept;
  [[nodiscard]] bool move_wins_board(int group, int cell,
                                     int8_t piece) const noexcept;
  [[nodiscard]] bool player_has_local_win_threat(
      int group, int8_t piece, int exclude_cell = -1,
      int8_t exclude_piece = 0) const noexcept;
  [[nodiscard]] bool move_sends_to_local_win_threat(
      int group, int cell, int8_t piece,
      int8_t opponent_piece) const noexcept;
  [[nodiscard]] bool move_gives_opponent_free_choice(
      int group, int cell, int8_t piece) const noexcept;
  static void group_cell_to_physical(int group, int cell, int* row,
                                     int* col) noexcept;
  static void physical_to_group_cell(int row, int col, int* group,
                                     int* cell) noexcept;
  static std::pair<int, int> rotate_coord(int row, int col,
                                          int turns) noexcept;
  static std::pair<int, int> transform_coord(int row, int col, int turns,
                                             bool flip) noexcept;
  static uint32_t transform_action(uint32_t action, int turns,
                                   bool flip) noexcept;

  BoardTensor board_{};
  int8_t player_{0};
  uint32_t turn_{0};
};

}  // namespace alphazero::uttt_gs
