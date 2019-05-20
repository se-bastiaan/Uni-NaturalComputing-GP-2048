import random
from functools import partial

import numpy as np
from deap import base, creator, gp, tools, algorithms

from game import Game


def progn(*args):
    for arg in args:
        arg()


def prog2(out1, out2):
    return partial(progn, out1, out2)


def prog3(out1, out2, out3):
    return partial(progn, out1, out2, out3)


def if_then_else(condition, out1, out2):
    out1() if condition() else out2()


class GamePlayer:
    game = Game()

    def _reset(self):
        self.game = Game()

    def fitness(self):
        return np.sum(self.game.matrix) + 9 * self.game.highest_tile()

    def if_lost(self, out1, out2):
        return partial(if_then_else, self.game.has_lost, out1, out2)

    def up(self):
        self.game.up()

    def down(self):
        self.game.down()

    def left(self):
        self.game.left()

    def right(self):
        self.game.right()

    def play(self, routine):
        self._reset()
        last_score = 0
        steps = 0
        while not self.game.has_lost() and steps < 100:
            routine()
            if last_score == self.game.current_score():
                steps += 1
            else:
                steps = 0
            last_score = self.game.current_score()


player = GamePlayer()

pset = gp.PrimitiveSet("main", 0)
pset.addTerminal(player.up, name='up')
pset.addTerminal(player.down, name='down')
pset.addTerminal(player.left, name='left')
pset.addTerminal(player.right, name='right')
# pset.addPrimitive(player.if_lost, 2)
pset.addPrimitive(prog2, 2)
pset.addPrimitive(prog3, 3)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=2)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalGame(individual):
    # Transform the tree expression to functionnal Python code
    routine = gp.compile(individual, pset)
    # Run the generated routine
    player.play(routine)

    player.game.print_game()
    print(player.game.highest_tile(), player.fitness())

    return player.fitness(), player.game.highest_tile()


toolbox.register("evaluate", evalGame)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


def main():
    random.seed(69)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 40, stats, halloffame=hof, verbose=True)

    evalGame(hof.items[0])

    return pop, hof, stats


if __name__ == "__main__":
    main()