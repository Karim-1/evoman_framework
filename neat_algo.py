################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################


# imports framework & libraries
import sys, os
import neat
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

# import classes
sys.path.insert(0, 'evoman') 

from demo_controller import NEAT_controller
from environment import Environment
from play_winner import play_winner
from plot_functions import plot_fitness, plot_best

# TODO:
# --> may be good to crossect between individuals that are most different
# --> use around 25-150 generations (common in literature), roughly 2-5 changes per generation (depends on pop size)

# stop condition --> reaching some minimum level of diversity?
# or --> reaching some specified number of generations without fitness improvement


def sigma_scale(f):
    '''
    sigma scales the fitness according to the sigma scaling function
    a constant of 2 is used
    --> we did not use this function eventually
    '''
    mean_f = np.mean(f)
    sigma_f = np.std(f)
    c = 2
    
    for i in range(len(f)):
        f[i] = np.max(f[i]-(mean_f-c*sigma_f), 0)
    
    return f


def simulate(genome):
    return env.play(pcont=genome)


def eval_genomes(genomes, config):
    '''
    evaluates the fitness of each genome
    and stores the best and mean fitness
    '''
    fitnesses = []
    for genome_id, genome in genomes:
        fitness, p, e, t = simulate(genome)
        genome.fitness = fitness
        fitnesses.append(fitness)
    
    '''
    # sigma scale the fitnesses of the genomes
    scaled_fitness = sigma_scale(fitnesses)
    print(fitnesses)
    print(scaled_fitness)

    for i in range(len(genomes)):
        genomes[i][1].fitness = scaled_fitness[i]
    '''

    # # retrieve mean and best fitness for the genomes
    # mean_fitness.append(np.mean(fitnesses))
    # best_fitness.append(np.max(fitnesses))

    return genomes
    

# run algorithm
def run_neat(path, experiment_name):
    N_GENERATIONS = 25
    
    # create NEAT configuration
    config = neat.Config(neat.DefaultGenome, 
                        neat.DefaultReproduction, 
                        neat.DefaultSpeciesSet, 
                        neat.DefaultStagnation,
                        config_file)
    # create NEAT population
    pop = neat.Population(config)
    
    # add reporter for population
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    # # this can be commented out if you need a checkpointer
    # pop.add_reporter(neat.Checkpointer(10))
    
    t0 = time.time()

    # run all generations and pick winner
    winner = pop.run(eval_genomes, N_GENERATIONS)

    # Display the winning genome
    print('\nBest genome:\n{!s}'.format(winner))

    with open(f"results/{experiment_name}_results/winner.pkl", "wb") as f:
        pickle.dump(winner, f)
        f.close()

    # retrieve best and mean fitness from stats
    best_fitness = [c.fitness for c in stats.most_fit_genomes]
    mean_fitness = np.array(stats.get_fitness_mean())
    np.save(f"results/{experiment_name}_results/mean_fitness", mean_fitness)
    np.save(f"results/{experiment_name}_results/best_fitness", best_fitness)
    print('MEAN FITNESS:\n', mean_fitness)
    print('BEST FITNESS:\n', best_fitness)

    # report time simulation took
    t1 = time.time()
    print(f'\n----- simulation took {np.round((t1-t0)/60,1)} minutes -----\n')

    # plot results
    plot_fitness(mean_fitness, best_fitness, experiment_name)


if __name__ == "__main__":
    experiment_name = 'partial_nodirect_pop50'
    enemies = [2, 4, 5]

    if not os.path.exists(f'results/{experiment_name}_results'):
        os.makedirs(f'results/{experiment_name}_results')
    
    # retrieve configuration file
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat_config.txt")

    # remove game display
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    # initialize environment with NEAT network as player controller
    env = Environment(
        experiment_name=experiment_name,
        enemies=enemies,
        playermode = 'ai',
        enemymode='static',
        sound='off',
        player_controller=NEAT_controller(),
        level=2,
        speed='fastest',
        logs='off'
        )

    # lists to store information for each generation
    mean_fitness = []
    best_fitness = []

    # run experiments
    run_neat(config_file, experiment_name)