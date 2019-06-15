import numpy as np

import logic
from constants import GRID_LEN, Move
from player import DumbPlayer


class Game:
    matrix = None
    commands = {
        Move.Left: logic.left,
        Move.Right: logic.right,
        Move.Up: logic.up,
        Move.Down: logic.down,
    }

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
                row_line += ' {}  '.format(value)
            print(row_line)

    def current_score(self):
        return logic.game_score(self.matrix)

    def fitness(self):
        return np.sum(self.matrix) + 9 * self.highest_tile()

    def has_lost(self):
        return logic.game_state(self.matrix) == logic.STATE_LOSE

    def highest_tile(self):
        score = 0
        for row, values in enumerate(self.matrix):
            for col, x in enumerate(values):
                if self.matrix[row][col] > score:
                    score = self.matrix[row][col]
        return score

    def _process_move(self, move):
        try:
            new_state, done = self.commands[move](self.matrix)
            if done:  # done = not a fully filled game matrix
                self.matrix = logic.add_tile(self.matrix)
            self.matrix = new_state
        except KeyError:
            raise Exception("Move is not one of Left, Up, Right, Down but is {}".format(str(move)))

    def play_game(self, player):
        self.new_game()

        loop_detected = False
        while not loop_detected:
            previous_score = logic.total_value(self.matrix)

            move = player.play(self.matrix)
            self._process_move(move)
            self.matrix = logic.add_tile(self.matrix)

            new_score = logic.total_value(self.matrix)
            loop_detected = new_score == previous_score  # If there is no increase in the score, then the move did nothing

        return logic.total_value(self.matrix), self.highest_tile(), self.fitness()


if __name__ == '__main__':
    print("Let's play a game")
    player = DumbPlayer()

    automated_game = Game()
    score = automated_game.play_game(player)

    print("You got a total value of {}".format(score))
