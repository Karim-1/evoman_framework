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

import time
import numpy as np
from math import fabs,sqrt
import glob, os

from deap import base
from deap import creator
from deap import tools

import random
from itertools import repeat
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'SGA'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  level=2,
                  playermode="ai",
                  enemymode="static",
                  speed="fastest",
                  randomini="yes",
                  player_controller=player_controller(n_hidden_neurons),
                  #enemy_controller=enemy_controller(n_hidden_neurons)
                  )

start = time.time()

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

#initialize other variables
npop = 100
max_gens = 25
mut_prob = 0.1
LB = -1
UB = 1
cross_prob = 0.4

def simulate(genome):
    #Simulate one individual
    f,p,e,t = env.play(pcont = genome)
    return f
    
def selection(population):
    #Select best individual among tournsize groups, k times
    return tools.selTournament(pop = population, k = npop, tournsize = 10, fit_attr='fitness')

def mutUniformFloat(individual, low, up, indpb):
    #Adapted from mutUniformInt from the deap library
    #https://github.com/DEAP/deap/blob/master/deap/tools/mutation.py
    """Mutate an individual by replacing attributes, with probability *indpb*,
    by a integer uniformly drawn between *low* and *up* inclusively.
    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param low: The lower bound or a :term:`python:sequence` of
                of lower bounds of the range from which to draw the new
                float.
    :param up: The upper bound or a :term:`python:sequence` of
               of upper bounds of the range from which to draw the new
               float.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    size = len(individual)
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))

    for i, xl, xu in zip(range(size), low, up):
        if random.random() < indpb:
            individual[i] = random.uniform(xl, xu)

    return individual
    
#Create Deap functions
toolbox = base.Toolbox()
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("attr_uni", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_uni, n_vars)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", simulate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutUniformFloat, low=LB, up=UB, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    pop = toolbox.population(n=npop)
    fitnesses = list(map(toolbox.evaluate, pop))
    for individual, fit in zip(pop, fitnesses):
        individual.fitness.values = (fit,)
    
    for i in range(max_gens):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))    
        
         # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cross_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)
            
        pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in pop]
        
        
        best = np.argmax(fits)
        std  =  np.std(fits)
        mean = np.mean(fits)
    
        # saves results
        file_aux  = open(experiment_name+'/results.txt','a')
        print( '\n GENERATION '+str(i)+' '+str(round(fits[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
        file_aux.write('\n'+str(i)+' '+str(round(fits[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
        file_aux.close()
    
        # saves generation number
        file_aux  = open(experiment_name+'/gen.txt','w')
        file_aux.write(str(i))
        file_aux.close()
    
        # saves file with the best solution
        np.savetxt(experiment_name+'/best.txt',pop[best])
    
        # saves simulation state
        solutions = [pop, fits]
        env.update_solutions(solutions)
        env.save_state()
        

    end = time.time() # prints total execution time for experiment
    print( '\nExecution time: '+str(round((end-start)/60))+' minutes \n')
    
    
    file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
    file.close()
    
    
    env.state_to_log() # checks environment state
    
main()