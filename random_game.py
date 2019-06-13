import random

import numpy as np

from constants import Move
from game import Game


class RandomPlayer:
    def play(self, game):
        ret = random.sample([Move.Down, Move.Up, Move.Right, Move.Left], 1)
        return ret[0]


GAMES_PER_INDIVIDUAL = 10
player = RandomPlayer()

game = Game()

scores = np.array([game.play_game(player) for i in
                   range(GAMES_PER_INDIVIDUAL)]).transpose()

ret_val = (np.median(scores[0]),  # total sum of tiles
           np.max(scores[1]),  # max tile
           np.median(scores[2]))  # fitness calc result

print(ret_val)
