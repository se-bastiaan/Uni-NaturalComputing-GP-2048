import multiprocessing
import random
from functools import partial

import numpy as np
from deap import base, creator, gp, tools, algorithms

from game import Game


def progn(*args):
    return_values = []
    for arg in args:
        if isinstance(arg, str):
            return_values.append(arg)
        else:
            return_values.append(arg())
    return ' '.join(return_values)


def prog2(out1, out2):
    return partial(progn, out1, out2)


def prog3(out1, out2, out3):
    return partial(progn, out1, out2, out3)


def if_then_else(condition, out1, out2):
    return out1() if condition() else out2()


class GamePlayer:
    UP = 'up'
    DOWN = 'down'
    LEFT = 'left'
    RIGHT = 'right'

    game = None
    commands = None

    def __init__(self) -> None:
        super().__init__()
        self._reset()

    def _reset(self):
        self.game = Game()
        self.commands = {
            self.UP: self.game.up,
            self.DOWN: self.game.down,
            self.LEFT: self.game.left,
            self.RIGHT: self.game.right,
        }

    def fitness(self):
        return np.sum(self.game.matrix) + 9 * self.game.highest_tile()

    def if_lost(self, out1, out2):
        return partial(if_then_else, self.game.has_lost, out1, out2)

    def play(self, routine):
        self._reset()
        last_score = 0
        steps = 0
        while not self.game.has_lost() and steps < 100:
            actions = routine()
            for action in actions.split(' '):
                self.commands[action]()
            if last_score == self.game.current_score():
                steps += 1
            else:
                steps = 0
            last_score = self.game.current_score()

pset = gp.PrimitiveSet("main", 0)
pset.addTerminal(GamePlayer.UP, name='up')
pset.addTerminal(GamePlayer.DOWN, name='down')
pset.addTerminal(GamePlayer.LEFT, name='left')
pset.addTerminal(GamePlayer.RIGHT, name='right')
# pset.addPrimitive(player.if_lost, 2)
pset.addPrimitive(prog2, 2)
pset.addPrimitive(prog3, 3)


creator.create("FitnessMax", base.Fitness, weights=(1.0, 0.5,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()


# @todo
# Depth outputs
# min=1, max=2, highest tile = 512
# min=1, max=5, highest tile = 1024

# Attribute generator
toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=5)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalGame(individual):
    # Transform the tree expression to functionnal Python code
    routine = gp.compile(individual, pset)
    # Run the generated routine

    scores = []
    highest_tiles = []
    for i in range(0, 10):
        player = GamePlayer()
        player.play(routine)
        highest_tiles.append(player.game.highest_tile())
        scores.append(player.fitness())

    return np.median(highest_tiles), np.median(scores)


toolbox.register("evaluate", evalGame)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=5)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# pool = multiprocessing.Pool()
# toolbox.register("map", pool.map)


def stats_f(ind):
    return ind.fitness.values


def stats_min(i):
    return list(map(np.min, zip(*i)))


def stats_max(i):
    return list(map(np.max, zip(*i)))


def stats_std(i):
    return list(map(np.std, zip(*i)))


def stats_avg(i):
    return list(map(np.mean, zip(*i)))


def main():
    random.seed(69)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(stats_f)
    stats.register("min", stats_min)
    stats.register("max", stats_max)
    stats.register("std", stats_std)
    stats.register("avg", stats_avg)

    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 40, stats, halloffame=hof, verbose=True)

    best = hof.items[0]
    evalGame(best)

    return pop, hof, stats


if __name__ == "__main__":
    main()
