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

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

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
    

def play_winner(env, winner):
        sum_gains = 0
        for i in range(5):
            f,p,e,t = env.play(pcont=winner)
            gain = p-e
            print(i,':',p,e)
            sum_gains += gain
        print(sum_gains/5)
        return sum_gains/5   


def plot_boxplot(data_ES,data_SGA, name):
    #enemynames = ['_', '_', 'Airman (enemy 2)', '_', 'HeatMan (enemy 4)', 'Metalman (enemy 5)']
    #colors = ['_', '_', 'blue', '_', 'red', 'black']
    #Make a boxplot of two arrays of data             
    data = [data_ES, data_SGA]
    
    fig, ax = plt.subplots(figsize =(6, 6))
    ax.set_title(f'Training group {name}', fontsize=18)#, color=colors[enemy])
    ax.boxplot(data)
    plt.ylabel('total gain', fontsize=18)
    ax.set_xticklabels(['ES', 'SGA'], fontsize=18)
    plt.savefig(f'plots2/boxplot_en{name}')
    plt.show()    
    
def my_cons_multi(values):
    return values.sum()

if __name__ == "__main__":
        
        n_hidden_neurons = 10
        enemies = [1,2,3,4,5,6,7,8]
        name = '[7,8]'
        
        if not os.path.exists(f'plots2/'):
            os.makedirs(f'plots2/')
        means_ES = []
        means_SGA = []
            
        ES_env = Environment(
                experiment_name=f'play_winner_ES_enemy{name}',
                multiplemode='yes',
                enemies=enemies,
                playermode = 'ai',
                enemymode='static',
                sound='off',
                player_controller=player_controller(n_hidden_neurons),
                level=2,
                speed='fastest',
                logs='off',
                randomini="yes"
                )
            
        SGA_env = Environment(
                    experiment_name=f'play_winner_SGA_enemy{name}',
                    multiplemode='yes',
                    enemies=enemies,
                    playermode = 'ai',
                    enemymode='static',
                    sound='off',
                    player_controller=player_controller(n_hidden_neurons),
                    level=2,
                    speed='fastest',
                    logs='off',
                    randomini="yes"
                    )    
        
        ES_env.cons_multi = my_cons_multi
        SGA_env.cons_multi = my_cons_multi
            
    
        for i in range(1,11,1):
            print(f'Experiment {i}:')
                
            genome_ES = np.load(f'results_ES/en{name}/overall_best{i-1}.npy')
            mean_ES = play_winner(ES_env, genome_ES)
            means_ES.append(mean_ES)
                
            genome_SGA = np.load(f'results_SGA2/en{name}-{i}/overall_best.npy')
            mean_SGA = play_winner(SGA_env, genome_SGA)
            means_SGA.append(mean_SGA)
        print(means_ES)
        print(means_SGA)
        plot_boxplot(means_ES, means_SGA, name)
            
        np.save(f'results_ES/final_experiment/en{name}_mean_gains.npy', means_ES)
        np.save(f'results_SGA2/final_experiment/en{name}_mean_gains.npy', means_SGA)