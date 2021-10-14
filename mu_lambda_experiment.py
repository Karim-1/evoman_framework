################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from SGA_controller import player_controller, enemy_controller

import numpy as np
import random
import time

from deap import algorithms
from deap import base
from deap import creator
from deap import tools


from itertools import repeat
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence



def evaluate(genome):
    # simulate one individual
    fitness,_,_,_ = env.play(pcont = genome)
    return fitness
    
def selection(population):
    # select best individual among tournsize groups, k times
    return tools.selTournament(pop = population, k = MU, tournsize = 10, fit_attr='fitness')


def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                        stats=None, halloffame=None, verbose=__debug__):
    '''
    copied from the DEAP library with minor adjustments (line 147 only)
    '''
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook
    

    
if __name__=="__main__":
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # create folder
    experiment_name = 'test'
    if not os.path.exists('results_ES/'+experiment_name):
        os.makedirs('results_ES/'+experiment_name)

    n_hidden_neurons = 10
    enemies = [7,8]

    # initializes environment with ai player using random controller, playing against static enemy
    env = Environment(experiment_name='results_ES/'+experiment_name,
                    multiplemode='yes',
                    enemies=enemies,
                    level=2,
                    playermode="ai",
                    enemymode="static",
                    speed="fastest",
                    randomini="yes",
                    player_controller=player_controller(n_hidden_neurons),
                    logs='off'
                    )

    def my_const_multi(values):
        return values.mean()
    
    env.cons_multi = my_const_multi
    
    t0 = time.time()

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    #initialize other variables
    MUs= range(25,101,25)
    LAMBDAs = range(25,101,25)
    
    SIGMA = 1.5
    ngen = 10
    mutpb = 0.2 # mutation probability
    LB = -1
    UB = 1
    cxpb = 0.4 # crossing probability

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    final_list = []
    for MU in MUs:
        max_fitnesses = []
        for LAMBDA in LAMBDAs:
            start_time = time.time()
            print(f'\n----- Running mu = {MU}, lambda = {LAMBDA} -----')    

            # create deap functions
            toolbox = base.Toolbox()
            hof = tools.ParetoFront()
            
            toolbox.register("attr_uni", random.uniform, -1, 1)
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_uni, n_vars)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", evaluate)
            toolbox.register("crossover", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutGaussian, mu=MU, sigma=SIGMA, indpb=0.05)
            toolbox.register("select", tools.selTournament, tournsize=3)
            toolbox.register("mate", tools.cxTwoPoint)
            
            # create statistics functions
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean, axis=0)
            stats.register("std", np.std, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)

            # create population
            pop = toolbox.population(n=MU)

            fitnesses = list(map(toolbox.evaluate, pop))
            for individual, fit in zip(pop, fitnesses):
                individual.fitness.values = (fit,)


            # best_fitness, mean_fitness = [], []
            population, logbook =  eaMuPlusLambda(pop, toolbox, MU, LAMBDA, cxpb, mutpb, ngen, stats)

            # print(population)
            record = stats.compile(pop)
            max_fitnesses.append(record['max'][0])

            end_time = time.time()
            print(f'Sim run took {round((end_time-start_time)/60, 2)} minutes')

            final_list.append(max_fitnesses)
            
    np.savetxt(f'results_ES/{experiment_name}/max_fitnesses.csv', final_list, delimiter=',')


    t1 = time.time()

    print(f'\n---------- Simulation took {round((t1-t0)/60, 2)} minutes-------------------')

