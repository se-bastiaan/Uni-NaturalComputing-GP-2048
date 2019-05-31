import multiprocessing
import operator
import random
from functools import partial

import numpy as np
from deap import base, creator, gp, tools, algorithms

# import logic
from constants import Move
from game import Game
from player import GPPlayer


def progn(*args):
    for arg in args:
        arg()


def prog2(out1, out2):
    return partial(progn, out1, out2)


def prog3(out1, out2, out3):
    return partial(progn, out1, out2, out3)


def if_then_else(condition, out1, out2):
    return out1 if condition else out2


# class GamePlayer:
#     game = Game()

#     def _reset(self):
#         self.game = Game()

#     def fitness(self):
#         return sum(self.game.matrix) + 9 * self.game.highest_tile()

#     def if_lost(self, out1, out2):
#         return partial(if_then_else, self.game.has_lost, out1, out2)

#     def up(self):
#         self.game.up()

#     def down(self):
#         self.game.down()

#     def left(self):
#         self.game.left()

#     def right(self):
#         self.game.right()

#     def play(self, routine):
#         self._reset()
#         while not self.game.has_lost():
#             routine()


# player = GamePlayer()

# Routines we can give the player:
# * Get position of highest tile
# * (integer) Constants




pset = gp.PrimitiveSetTyped("main", [int] * 16, str)
pset.addTerminal(Move.Left, str, name='left')
pset.addTerminal(Move.Up, str, name='up')
pset.addTerminal(Move.Right, str, name='right')
pset.addTerminal(Move.Down, str, name='down')

pset.addTerminal(True, bool, name='bool_true')

pset.addTerminal(12, int, name='value0')
pset.addPrimitive(if_then_else, [bool, str, str], str, name='if_then_else')
pset.addPrimitive(operator.eq, [int, int], bool, name='eq')
pset.addPrimitive(operator.lt, [int, int], bool, name='lt')
pset.addPrimitive(operator.gt, [int, int], bool, name='gt')


GAMES_PER_INDIVIDUAL = 10
def evaluateIndividual(individual):
    fn = gp.compile(individual, pset)
    player = GPPlayer(fn)
    game = Game()

    scores = [game.play_game(player) for i in range(GAMES_PER_INDIVIDUAL)]

    return tuple(map(np.median, list(zip(*scores))))


creator.create("FitnessMax", base.Fitness, weights=(0.1, 1.0, 0.5))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=5)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


toolbox.register("evaluate", evaluateIndividual)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

pool = multiprocessing.Pool()
toolbox.register("map", pool.map)


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
    # random.seed(37)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(stats_f)
    stats.register("min", stats_min)
    stats.register("max", stats_max)
    stats.register("std", stats_std)
    stats.register("avg", stats_avg)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hof)

    return pop, hof, stats


if __name__ == "__main__":
    main()