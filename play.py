from enum import Enum, unique

import logic
from constants import *
import readchar


@unique
class Terminal(Enum):
    Left = 1
    Up = 2
    Right = 3
    Down = 4


class Player(object):
    def __init__(self):
        pass
    
    def play(self, game):
        raise NotImplementedError('Implement this in a subclass of Player!')


class DumbPlayer(object):
    did_left = False

    def play(self, game):
        if self.did_left:
            self.did_left = False
            return Terminal.Down
        else:
            self.did_left = True
            return Terminal.Left


class Automated2048(object):
    state = None

    def __init__(self):
        pass

    def play_game(self, player):
        self._new_game()

        loop_detected = False
        while not loop_detected:
            previous_score = logic.total_value(self.state)

            move = player.play(self.state)
            self.process_move(move)
            self.state = logic.add_tile(self.state)

            new_score = logic.total_value(self.state)
            loop_detected = new_score == previous_score  # If there is no increase in the score, then the move did nothing
        
        return logic.total_value(self.state)

    def _new_game(self):
        self.state = logic.new_game(GRID_LEN)
        self.state = logic.add_tile(self.state)
        self.state = logic.add_tile(self.state)

    def process_move(self, move):
        if move == Terminal.Left:
            new_state, _ = logic.left(self.state)
        elif move == Terminal.Up:
            new_state, _ = logic.up(self.state)
        elif move == Terminal.Right:
            new_state, _ = logic.right(self.state)
        elif move == Terminal.Down:
            new_state, _ = logic.down(self.state)
        
        self.state = new_state

    def print_game(self):
        for row in self.state:
            row_line = ''
            for value in row:
                row_line += f' {value}  '
            print(row_line)


if __name__ == '__main__':
    print("Let's play a game")
    player = DumbPlayer()

    automated_game = Automated2048()
    score = automated_game.play_game(player)

    print("You got a total value of {}".format(score))