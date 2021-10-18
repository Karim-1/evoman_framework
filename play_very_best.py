'''
Can be used to show gameplay of a the very best genome
'''

import pickle
import numpy as np
import sys, os
sys.path.insert(0, 'evoman') 

#from demo_controller import NEAT_controller
from SGA_controller import player_controller
from environment import Environment

#from plot_functions import plot_boxplot

import matplotlib.pyplot as plt

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    

def play_winner(env, winner):
        sum_p = 0
        sum_e = 0
        for i in range(5):
            f,p,e,t = env.play(pcont=winner)
            sum_p += p
            sum_e += e
        print(sum_p/5, sum_e/5)
        return sum_p/5, sum_e/5

if __name__ == "__main__":
        
        n_hidden_neurons = 10
        enemies = [1,2,3,4,5,6,7,8]
        name = 'very_best_results'
        
        #if not os.path.exists(f'very_best_results'):
        #    os.makedirs(f'plots2/')
            
        
        very_best_genome = np.load(f'results_SGA2/en[2,5,6]-3/overall_best.npy')
        for en in enemies:
            
            env = Environment(
                    experiment_name=f'{name}',
                    enemies=[en],
                    playermode = 'ai',
                    enemymode='static',
                    sound='off',
                    player_controller=player_controller(n_hidden_neurons),
                    level=2,
                    speed='fastest',
                    logs='off',
                    randomini="yes"
                    )  
            
            print(f'Enemy {en}:')
            
            player_energy, enemy_energy = play_winner(env, very_best_genome)
            # saves results
            file_aux  = open(f'{name}2.txt','a')
            file_aux.write(f'enemy {en}: player energy = {player_energy}, enemy energy = {enemy_energy}\n')
            file_aux.close()