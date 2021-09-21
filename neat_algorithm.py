# imports framework & libraries
import sys, os
import neat
import numpy as np

# import classes
from demo_controller import player_controller, enemy_controller


sys.path.insert(0, 'evoman') 
from environment import Environment

experiment_name = 'neat_algorithm'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name)
env.play()

# test
