import logic
from constants import *
import readchar


class Terminal2048:
    game_matrix = None
    commands = []

    def __init__(self) -> None:
        super().__init__()

        self.commands = {
            readchar.key.UP: logic.up,
            readchar.key.DOWN: logic.down,
            readchar.key.LEFT: logic.left,
            readchar.key.RIGHT: logic.right,
        }

        print('Press <q> to close, <r> for reset, <s> to print score. '
              'Arrow keys for controls.')

        self.new_game()
        self.main_loop()

    def new_game(self):
        self.game_matrix = logic.new_game(GRID_LEN)
        self.game_matrix = logic.add_tile(self.game_matrix)
        self.game_matrix = logic.add_tile(self.game_matrix)
        self.print_game()

    def print_score(self):
        print(f'Current score: {logic.game_score(self.game_matrix)}\n')

    def print_game(self):
        for row in self.game_matrix:
            row_line = ''
            for value in row:
                row_line += f' {value}  '
            print(row_line)
        self.print_score()

    def process_move(self, key):
        if logic.game_state(self.game_matrix) == logic.STATE_PROGRESS:
            if key not in self.commands:
                return
            self.game_matrix, done = self.commands[key](self.game_matrix)
            if done:  # done = not a fully filled game matrix
                self.game_matrix = logic.add_tile(self.game_matrix)
            self.print_game()
            if logic.game_state(self.game_matrix) == logic.STATE_WIN:
                print('You have won! Press <r> to reset, <q> to close.\n')
            if logic.game_state(self.game_matrix) == logic.STATE_LOSE:
                print('You lost! Press <r> to reset, <q> to close.\n')

    def main_loop(self):
        while True:
            key = readchar.readkey()
            if key == 'q':
                exit()
            elif key == 'r':
                self.new_game()
            elif key == 's':
                self.print_score()
            else:
                self.process_move(key)


terminal = Terminal2048()
