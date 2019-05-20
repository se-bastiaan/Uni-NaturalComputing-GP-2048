import logic
from constants import GRID_LEN


class Game:
    matrix = None
    commands = []

    def __init__(self) -> None:
        super().__init__()

        self.new_game()

    def new_game(self):
        self.matrix = logic.new_game(GRID_LEN)
        self.matrix = logic.add_tile(self.matrix)
        self.matrix = logic.add_tile(self.matrix)

    def print_game(self):
        for row in self.matrix:
            row_line = ''
            for value in row:
                row_line += f' {value}  '
            print(row_line)

    def current_score(self):
        return logic.game_score(self.matrix)

    def has_lost(self):
        return logic.game_state(self.matrix) == logic.STATE_LOSE

    def highest_tile(self):
        score = 0
        for row, values in enumerate(self.matrix):
            for col, x in enumerate(values):
                if self.matrix[row][col] > score:
                    score = self.matrix[row][col]
        return score

    def _process_move(self, action):
        if logic.game_state(self.matrix) == logic.STATE_PROGRESS:
            self.matrix, done = action(self.matrix)
            if done:  # done = not a fully filled game matrix
                self.matrix = logic.add_tile(self.matrix)

    def up(self):
        self._process_move(logic.up)

    def down(self):
        self._process_move(logic.down)

    def left(self):
        self._process_move(logic.left)

    def right(self):
        self._process_move(logic.right)
