################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################


# imports framework & libraries
import sys, os
import neat
import numpy as np

# import classes
sys.path.insert(0, 'evoman') 

from controller import Controller
from demo_controller import player_controller, enemy_controller
from environment import Environment

# run algorithm
def run(path, experiment_name):
    # create NEAT configuration
    config = neat.Config(neat.DefaultGenome, 
                        neat.DefaultReproduction, 
                        neat.DefaultSpeciesSet, 
                        neat.DefaultStagnation,
                        config_file)
    
    # create random NEAT population
    p = neat.Population(config)

    # initializes environment with ai player using random controller, playing against static enemy
    env = Environment(
        experiment_name=experiment_name
        )
    # env.play()
    print(env.play())

    # stop condition --> reaching some minimum level of diversity?
    # or --> reaching some specified number of generations without fitness improvement

# test
if __name__ == "__main__":
    experiment_name = 'neat_algorithm'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat-config.txt")
    run(config_file, experiment_name)