################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################


# imports framework & libraries
import sys, os
import neat
import numpy as np
import time

# import classes
sys.path.insert(0, 'evoman') 

from controller import Controller
from demo_controller import NEAT_controller
from environment import Environment


def eval_genomes(genomes, config):
    '''
    evaluates the fitness of each genome
    '''
    for genome_id, genome in genomes:
        fitness, plife, elife, sim_time = simulate(genome)
        genome.fitness = fitness
        print('\n\n genome fitness=',genome.fitness)

    return genomes
    
    
# --> may be good to crossect between individuals that are most different
# --> use around 25-150 generations (common in literature), roughly 2-5 changes per generation (depends on pop size)

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
    
    # create random NEAT population
    pop = neat.Population(config)
    
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(5))
    
    winner = pop.run(eval_genomes,50)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    
    t0 = time.time()
    
    
    
    ''' stop condition --> reaching some minimum level of diversity?
    or --> reaching some specified number of generations without fitness improvement '''
    
    t1 = time.time()
    
    print(f'\n----- simulation took {t1-t0} seconds -----\n')


if __name__ == "__main__":
    experiment_name = 'neat'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    
    # retrieve configuration file
    local_dir = os.path.dirname(__file__)
    if experiment_name == 'neat':
        config_file = os.path.join(local_dir, "neat_config.txt")

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
    run(config_file, experiment_name)