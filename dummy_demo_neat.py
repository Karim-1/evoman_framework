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


experiment_name = 'neat_algorithm'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(
    experiment_name=experiment_name,
    # player_controller=player_controller(Controller),
    # enemy_controller=enemy_controller()
)
env.play()


# run algorithm
def run(path):
    config = neat.config.Config(fitness_criterion = max,
     neat.DefaultGenome,
     neat.DefaultReproduction, 
     neat.DefaultSpeciesSet, 
     neat.DefaultStagnation, 
     path)

# test
if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)