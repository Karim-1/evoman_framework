'''
Can be used to show gameplay of a certain genome
'''

import pickle
import sys, os
sys.path.insert(0, 'evoman') 

from demo_controller import NEAT_controller
from environment import Environment


def play_winner(env, winner):

        # change genome structure so it can be used to play
        genomes = [(1, winner)]

        for genome_id, g in genomes:
                env.play(pcont=g)


if __name__ == "__main__":
        env = Environment(
                experiment_name='play_winner',
                enemies=[4],
                playermode = 'ai',
                enemymode='static',
                sound='off',
                player_controller=NEAT_controller(),
                level=2,
                speed='normal',
                logs='off'
                )

        with open('results/population_experiment/undirected_pop100_en4/winner.pkl', "rb") as f:
                genome = pickle.load(f)

        play_winner(env, genome)