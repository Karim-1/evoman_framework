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
import time

# import classes
sys.path.insert(0, 'evoman') 

from controller import Controller
from demo_controller import NEAT_controller
from environment import Environment

# TODO:
# --> may be good to crossect between individuals that are most different
# --> use around 25-150 generations (common in literature), roughly 2-5 changes per generation (depends on pop size)
# hoi Jim


def eval_genomes(genomes, config):
    '''
    evaluates the fitness of each genome
    and stores the best and mean fitness
    '''
    genome_fitnesses = []
    for genome_id, genome in genomes:
        fitness, p, e, t = simulate(genome)

        genome.fitness = fitness
        genome_fitnesses.append(fitness)
    
    # retrieve mean and best fitness for the genomes
    mean_fitness.append(np.mean(genome_fitnesses))
    best_fitness.append(np.max(genome_fitnesses))


    return genomes
    

def simulate(genome):
    return env.play(pcont=genome)

# run algorithm
def run(path, experiment_name):
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
    pop.add_reporter(neat.Checkpointer(5))
    
    t0 = time.time()

    # run all generations and pick winner
    winner = pop.run(eval_genomes,10)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    
    ''' stop condition --> reaching some minimum level of diversity?
    or --> reaching some specified number of generations without fitness improvement '''
    
    t1 = time.time()

    plt.scatter(range(1, len(mean_fitness)+1, 1), mean_fitness, label='Mean genome fitness')
    plt.scatter(range(1, len(best_fitness)+1, 1), best_fitness, label='Best genome fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()
    
    print(f'\n----- simulation took {np.round(t1-t0,1)} seconds -----\n')
    print(mean_fitness)
    print(best_fitness)


if __name__ == "__main__":
    experiment_name = 'neat'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    
    # retrieve configuration file
    local_dir = os.path.dirname(__file__)
    if experiment_name == 'neat':
        config_file = os.path.join(local_dir, "neat_config.txt")

    # remove game display
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    # initialize environment with NEAT network as player controller
    env = Environment(
        experiment_name=experiment_name,
        enemies=[2],
        playermode = 'ai',
        enemymode='static',
        sound='off',
        player_controller=NEAT_controller(),
        level=2,
        speed='fastest'
        )

    # lists to store information for each generation
    mean_fitness = []
    best_fitness = []

    # run experiments
    run(config_file, experiment_name)