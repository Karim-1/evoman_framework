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


import csv
import numpy as np
import random
import time

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from objproxies import CallbackProxy

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
    return tools.selTournament(pop = population, k = LAMBDA, tournsize = 10, fit_attr='fitness')



def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                        stats=None, halloffame=None, verbose=__debug__):
    '''
    copied from the DEAP library with minor adjustments (line 147 only)
    '''
    global gen_nr
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
        offspring = algorithms.varOr(population, toolbox, mu, cxpb, mutpb)

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
            
        gen_nr += 1

    return population, logbook

def my_const_multi(values):
    return values.mean()

    
if __name__=="__main__":
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # create folder
    experiment_name ='en[2,5,6]'
    if not os.path.exists('results_ES/'+experiment_name):
        os.makedirs('results_ES/'+experiment_name)

    n_hidden_neurons = 10
    enemies = [2,5,6]

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
    
    env.cons_multi = my_const_multi
    
    t0 = time.time()

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    #initialize other variables
    LAMBDA = 75 # population size
    MU = 75 # offspring
    cxpb = 0.4 # crossing probability
    mutpb = 0.2 # mutation probability
    LB = -1
    UB = 1    
    SIGMA = 1.5 # for the gaussian distribution
    ngen = 20
    gen_nr = 1 # define generation nr for mutation probability
    

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax) 

    # create deap functions
    toolbox = base.Toolbox()
    
    # register toolbox function
    toolbox.register("attr_uni", random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_uni, n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("crossover", tools.cxTwoPoint)
    # toolbox.register("mutate", tools.mutGaussian, mu=MU, sigma=SIGMA, indpb=CallbackProxy(lambda: .9**gen_nr))
    toolbox.register("mutate", tools.mutGaussian, mu=MU, sigma=SIGMA, indpb=.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxTwoPoint)
    
    # create statistics functions
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    overall_best = 0

    for i in range(10):   
        print(f'\n----- Running mu = {MU}, lambda = {LAMBDA}, ngen = {ngen}, round = {i} -----')   
        #  create population
        pop = toolbox.population(n=LAMBDA)

        fitnesses = list(map(toolbox.evaluate, pop))
    
        for individual, fit in zip(pop, fitnesses):
            individual.fitness.values = (fit,)

        # fits = [ind.fitness.values[0] for ind in pop]

        pop, log =  eaMuPlusLambda(pop, toolbox, MU, LAMBDA, cxpb, mutpb, ngen, stats)

        best = np.argmax(fitnesses)
        record = stats.compile(pop)

        max_fit = record['max'][0]
        print(max_fit)
        if max_fit > overall_best:
        # if fitnesses[best] > overall_best:
            overall_best = max_fit
            fits = [round(x,2) for x in fitnesses]
            best = np.where(fits == round(max_fit,2))
            print(best[0])
            best_genome = pop[best[0][0]]

        np.savetxt('results_ES/'+experiment_name+'/best.txt',best_genome)
        np.save('results_ES/'+experiment_name+'/overall_best', best_genome)
    
    np.savetxt('results_ES/'+experiment_name+'/best.txt',best_genome)

    t1 = time.time()

    print(f'\n------------------- Simulation took {round((t1-t0)/60, 2)} minutes -------------------')
    
    env.state_to_log()
    

