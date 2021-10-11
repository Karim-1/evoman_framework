'''
Can be used to show gameplay of a certain genome
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


#def play_winner_neat(env, winner):
#        
#        sum_gains = 0
#        
#        # change genome structure so it can be used to play
#        genomes = [(1, winner)]

#        for genome_id, g in genomes:
#            for i in range(5):            
#                f,p,e,t = env.play(pcont=g)
#                gain = p-e
#                print(i,':',p,e)
#                sum_gains += gain
#        print(sum_gains/5)
#        return sum_gains/5  
    

def play_winner_SGA(env, winner):
        sum_gains = 0
        for i in range(5):
            f,p,e,t = env.play(pcont=winner)
            gain = p-e
            print(i,':',p,e)
            sum_gains += gain
        print(sum_gains/5)
        return sum_gains/5   


def plot_boxplot(data_SGA, enemy):
    #enemynames = ['_', '_', 'Airman (enemy 2)', '_', 'HeatMan (enemy 4)', 'Metalman (enemy 5)']
    #colors = ['_', '_', 'blue', '_', 'red', 'black']
    #Make a boxplot of two arrays of data             
    data = [data_SGA]
    
    fig, ax = plt.subplots(figsize =(6, 6))
    ax.set_title(f'enemy {enemy}', fontsize=18)#, color=colors[enemy])
    ax.boxplot(data)
    plt.ylabel('individual gain', fontsize=18)
    #ax.set_xticklabels(['NEAT', 'SGA'], fontsize=18)
    plt.savefig(f'results_SGA2/plots/{name}/boxplot_en{enemy}')
    plt.show()    

if __name__ == "__main__":
        
        n_hidden_neurons = 10
        enemies = [1,2,3,4,5,6,7,8]
        name = 'en[2,5,6]_min'
        
        if not os.path.exists(f'results_SGA2/plots/{name}'):
            os.makedirs(f'results_SGA2/plots/{name}')
        
        for en in enemies:
            means_NEAT = []
            means_SGA = []
            
#            NEAT_env = Environment(
#                experiment_name=f'play_winner_NEAT_enemy{en}',
#                enemies=[en],
#                playermode = 'ai',
#                enemymode='static',
#                sound='off',
#                player_controller=NEAT_controller(),
#                level=2,
#                speed='fastest',
#                logs='off'
#                )
            
            SGA_env = Environment(
                    experiment_name=f'play_winner_SGA_enemy{en}',
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
            
    
            for i in range(1,11,1):
                print(f'enemy {en}, experiment {i}:')
                
                #with open(f'results_NEAT/final_experiment/final_enemy{en}_{i-1}/winner.pkl', "rb") as f:
                    #genome_NEAT = pickle.load(f)

                #mean_NEAT = play_winner_neat(NEAT_env, genome_NEAT)
                #means_NEAT.append(mean_NEAT)
                
                genome_SGA = np.load(f'results_SGA2/{name}/overall_best.npy')
                mean_SGA = play_winner_SGA(SGA_env, genome_SGA)
                means_SGA.append(mean_SGA)
            print(means_NEAT)
            print(means_SGA)
            plot_boxplot(means_SGA, en)
            
            #np.save(f'results_NEAT/final_experiment/enemy{en}_mean_gains.npy', means_NEAT)
            np.save(f'results_SGA2/{name}/e{en}_mean_gains.npy', means_SGA)