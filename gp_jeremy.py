import operator
import random
import multiprocessing
from functools import partial

import numpy
from deap import base, creator, gp, tools, algorithms


from game import Game
from play import Automated2048, GPPlayer




# Settings
GAMES_PER_INDIVIDUAL = 5
N_GENERATIONS = 40
N_INDIVIDUALS = 300

MAX_DEPTH = 5



# Routines we can give the player:
# * Get position of highest tile
# * (integer) Constants


def if_then_else(condition, out1, out2):
    return out1 if condition else out2

pset = gp.PrimitiveSetTyped("main", [int] * 16, str)
pset.addTerminal('left', str, name='left')
pset.addTerminal('up', str, name='up')
pset.addTerminal('right', str, name='right')
pset.addTerminal('down', str, name='down')

pset.addTerminal(True, bool, name='bool_true')

pset.addTerminal(12, int, name='value0')
pset.addPrimitive(if_then_else, [bool, str, str], str, name='if_then_else')
pset.addPrimitive(operator.eq, [int, int], bool, name='eq')
pset.addPrimitive(operator.lt, [int, int], bool, name='lt')
pset.addPrimitive(operator.gt, [int, int], bool, name='gt')



def evaluateIndividual(individual):
    fn = gp.compile(individual, pset)
    player = GPPlayer(fn)
    game = Automated2048()

    scores = [game.play_game(player) for i in range(GAMES_PER_INDIVIDUAL)]
    average = sum(scores) / GAMES_PER_INDIVIDUAL

    return [average]


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
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

def main():
    # random.seed(37)

    pop = toolbox.population(n=N_INDIVIDUALS)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.2, mutpb=0.5, ngen=N_GENERATIONS, stats=stats, halloffame=hof)

    return pop, hof, stats


if __name__ == "__main__":
    main()