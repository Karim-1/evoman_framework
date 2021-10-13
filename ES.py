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
    return tools.selTournament(pop = population, k = MU, tournsize = 10, fit_attr='fitness')



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
            
        gen_nr += 1

    return population, logbook
    

    
if __name__=="__main__":
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # create folder
    experiment_name = 'mutpb_experiment_gradual'
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

    def my_const_multi(values):
        return values.mean()
    
    env.cons_multi = my_const_multi
    
    t0 = time.time()

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    #initialize other variables
    MU= 50
    LAMBDA = 25
    
    SIGMA = 1.5
    ngen = 20

    gen_nr = 1 # define generation nr for mutation probability
    evals = 0 # define nr of evals to count gen number

    mutpb = 0.2 # mutation probability
    LB = -1
    UB = 1
    cxpb = 0.4 # crossing probability

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    print(f'\n----- Running mu = {MU}, lambda = {LAMBDA}, ngen = {ngen} -----')    

    # create deap functions
    toolbox = base.Toolbox()
    
    # register toolbox function
    toolbox.register("attr_uni", random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_uni, n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("crossover", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=MU, sigma=SIGMA, indpb=CallbackProxy(lambda: .9**gen_nr))
    # toolbox.register("mutate", tools.mutGaussian, mu=MU, sigma=SIGMA, indpb=.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxTwoPoint)
    
    # create statistics functions
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # create population
    pop = toolbox.population(n=LAMBDA)

    fitnesses = list(map(toolbox.evaluate, pop))
    
    for individual, fit in zip(pop, fitnesses):
        individual.fitness.values = (fit,)


    # best_fitness, mean_fitness = [], []
    population, log =  eaMuPlusLambda(pop, toolbox, MU, LAMBDA, cxpb, mutpb, ngen, stats)

    # print(population)
    print(log.select("avg"))
    print(log.select("max"))
    np.savez(f'results_ES/f{experiment_name}', name1=log.select("avg"), name2=log.select("max"), name3=log.select("std"))

    t1 = time.time()

    print(f'\n------------------- Simulation took {round((t1-t0)/60, 2)} minutes -------------------')

    # csvfile = f'results_ES/{experiment_name}/results.csv'
    # csv_columns = record.keys()
    # with open(csvfile) as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    #     # writer.writeheader()
    #     for key in record:
    #         writer.writerow(record[key])
    
    env.state_to_log()
    
    #     # saves results
    #     file_aux  = open('results_SGA2/'+experiment_name+'/results_SGA2.txt','a')
    #     print( '\n GENERATION '+str(i)+' '+str(round(fits[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    #     file_aux.write('\n'+str(i)+' '+str(round(fits[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    #     file_aux.close()
    
    #     # saves generation number
    #     file_aux  = open('results_SGA2/'+experiment_name+'/gen.txt','w')
    #     file_aux.write(str(i))
    #     file_aux.close()
    
    #     # saves file with the best solution of this generation
    #     np.savetxt('results_SGA2/'+experiment_name+'/best.txt',pop[best])
    
    #     # saves simulation state
    #     solutions = [pop, fits]
    #     env.update_solutions(solutions)
    #     env.save_state()
        

    # end = time.time() # prints total execution time for experiment
    # print( '\nExecution time: '+str(round((end-start)/60))+' minutes \n')
    
    # plot_fitness(mean_fitness, best_fitness, 'results_SGA2/'+experiment_name+'/plot_'+experiment_name)
    
    # np.save('results_SGA2/'+experiment_name+'/mean_fitness', mean_fitness)
    # np.save('results_SGA2/'+experiment_name+'/best_fitness', best_fitness)
    
    # # saves file with the overall solution
    # np.savetxt('results_SGA2/'+experiment_name+'/overall_best.txt',best_genome)
    # np.save('results_SGA2/'+experiment_name+'/overall_best', best_genome)
    
    # file = open('results_SGA2/'+experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
    # file.close()
    
    
    # env.state_to_log() # checks environment state
