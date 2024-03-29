import multiprocessing
import operator
import random
from functools import partial

import numpy as np
from deap import base, creator, gp, tools, algorithms

# import logic
from constants import Move, GRID_LEN
from game import Game
from player import GPPlayer, GridItem


def if_then_else(condition, out1, out2):
    return out1 if condition else out2



# Settings
GAMES_PER_INDIVIDUAL = 10
N_GENERATIONS = 3
N_INDIVIDUALS = 300

MAX_DEPTH = 5

# Routines we can give the player:
# * Get position of highest tile
# * (integer) Constants

def grid_equals(g1, g2):
    return g1.value == g2.value


def grid_lt(g1, g2):
    return g1.value < g2.value


def grid_gt(g1, g2):
    return g1.value > g2.value


def is_neighbour(g1, g2):
    return g1.location == (g2.location - 1) or g1.location == (
            g2.location + 1) or g1.location == (
                   g2.location - GRID_LEN) or g1.location == (
                   g2.location + GRID_LEN)

def grid_to_grid(input):
    return input


pset = gp.PrimitiveSetTyped("main", [GridItem] * (GRID_LEN * GRID_LEN), str)
pset.addTerminal(Move.Left, str, name='left')
pset.addTerminal(Move.Up, str, name='up')
pset.addTerminal(Move.Right, str, name='right')
pset.addTerminal(Move.Down, str, name='down')

pset.addTerminal(1, bool, name='bool_true')

pset.addPrimitive(grid_to_grid, [GridItem], GridItem, name='grid_item')
pset.addPrimitive(if_then_else, [bool, str, str], str, name='if_then_else')
pset.addPrimitive(grid_equals, [GridItem, GridItem], bool, name='eq')
pset.addPrimitive(grid_lt, [GridItem, GridItem], bool, name='lt')
pset.addPrimitive(grid_gt, [GridItem, GridItem], bool, name='gt')
pset.addPrimitive(is_neighbour, [GridItem, GridItem], bool, 'is_neighbour')

GAMES_PER_INDIVIDUAL = 11


def evaluateIndividual(individual):
    fn = gp.compile(individual, pset)
    player = GPPlayer(fn)
    game = Game()

    scores = np.array([game.play_game(player) for i in
                       range(GAMES_PER_INDIVIDUAL)]).transpose()

    ret_val = (np.median(scores[0]),  # total sum of tiles
               np.max(scores[1]),  # max tile
               np.median(scores[2]))  # fitness calc result

    return ret_val


creator.create("FitnessMax", base.Fitness, weights=(0.2, 1.0, 0.2))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=MAX_DEPTH)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluateIndividual)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Allow multiprocessing
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

    pop = toolbox.population(n=N_INDIVIDUALS)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(stats_f)
    stats.register("min", stats_min)
    stats.register("max", stats_max)
    stats.register("std", stats_std)
    stats.register("avg", stats_avg)

    algorithms.eaSimple(pop, toolbox, cxpb=0.2, mutpb=0.5, ngen=N_GENERATIONS, stats=stats, halloffame=hof)

    return pop, hof, stats


if __name__ == "__main__":
    pop, hof, stats = main()
    print("Best individual (re-evaluated):")
    print(evaluateIndividual(hof[0]))
