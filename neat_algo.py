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
from plot_fitness import plot_fitness

# TODO:
# --> may be good to crossect between individuals that are most different
# --> use around 25-150 generations (common in literature), roughly 2-5 changes per generation (depends on pop size)

# stop condition --> reaching some minimum level of diversity?
# or --> reaching some specified number of generations without fitness improvement


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
    
    # sigma scale the fitnesses of the genomes
    scaled_fitness = sigma_scale(genome_fitnesses)
    for i in range(len(genomes):
        genomes[i][1].fitness = scaled_fitness[i]
    
    # retrieve mean and best fitness for the genomes
    mean_fitness.append(np.mean(genome_fitnesses))
    best_fitness.append(np.max(genome_fitnesses))

    np.save(f"{experiment_name}_results/mean_fitness", mean_fitness)
    np.save(f"{experiment_name}_results/best_fitness", best_fitness)

    return genomes
    

def sigma_scale(f):
    '''
    sigma scales the fitness according to the sigma scaling function
    a constant of 2 is used
    '''
    mean_f = np.mean(f)
    sigma_f = np.std(f)
    c = 2
    
    for i in range(len(f)):
        f[i] = np.max(f[i]-(mean_f-c*sigma_f), 0)
    
    return f
    

def simulate(genome):
    return env.play(pcont=genome)

# run algorithm
def run(path, experiment_name):
    N_GENERATIONS = 50
    
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
    pop.add_reporter(neat.Checkpointer(10))
    
    t0 = time.time()

    # run all generations and pick winner
    winner = pop.run(eval_genomes, N_GENERATIONS)

    # Display the winning genome
    print('\nBest genome:\n{!s}'.format(winner))

    with open(f"{experiment_name}_results/winner.pkl", "wb") as f:
        pickle.dump(winner, f)
        f.close()

    t1 = time.time()

    # mean_fitness = np.load(f"{experiment_name}_results/best_fitness")
    # best_fitness = np.load(f"{experiment_name}_results/best_fitness")

    print(f'\n----- simulation took {np.round(t1-t0,1)} seconds -----\n')
    print(mean_fitness)
    print(best_fitness)

    # plot results
    plot_fitness(mean_fitness, best_fitness)

    # shows simulation of the winning genome
    play_winner(env, winner)


if __name__ == "__main__":
    experiment_name = 'neat'
    enemies = [1,2,3]

    if not os.path.exists(f'{experiment_name}_results'):
        os.makedirs(f'{experiment_name}_results')
    
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
    run(config_file, experiment_name)