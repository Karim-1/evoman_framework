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

def play_winner(env, winner):
        sum_gains = 0
        for i in range(5):
            f,p,e,t = env.play(pcont=winner)
            gain = p-e
            print(i,':',p,e)
            sum_gains += gain
        print(sum_gains/5)
        return sum_gains/5   


def plot_boxplot(data_SGA, train_group):
    #enemynames = ['_', '_', 'Airman (enemy 2)', '_', 'HeatMan (enemy 4)', 'Metalman (enemy 5)']
    #colors = ['_', '_', 'blue', '_', 'red', 'black']
    #Make a boxplot of two arrays of data             
    data = [data_SGA]
    
    fig, ax = plt.subplots(figsize =(6, 6))
    ax.set_title(f'Training group {train_group}', fontsize=18)#, color=colors[enemy])
    ax.boxplot(data)
    plt.ylabel('individual gain', fontsize=18)
    #ax.set_xticklabels(['NEAT', 'SGA'], fontsize=18)
    plt.savefig(f'results_SGA2/plots/{name}/boxplot_en{enemy}')
    plt.show()    

if __name__ == "__main__":
        # choose this for not using visuals and thus making experiments faster
        headless = True
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        
        n_hidden_neurons = 10
        enemies = [1,2,3,4,5,6,7,8]
        name = 'en[2,5,6]'
        
        if not os.path.exists(f'results_SGA2/plots/{name}'):
            os.makedirs(f'results_SGA2/plots/{name}')
<<<<<<< HEAD
            
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
                    experiment_name=f'play_winner_SGA_enemy',
                    multiplemode='yes',
                    enemies=enemies,
=======
        
        for en in enemies:
            means_ES = []
            means_SGA = []
            
            # env = Environment(
            #         experiment_name=f'play_winner_enemy{en}',
            #         enemies=[en],
            #         playermode = 'ai',
            #         enemymode='static',
            #         sound='off',
            #         player_controller=player_controller(n_hidden_neurons),
            #         level=2,
            #         speed='fastest',
            #         logs='off',
            #         randomini="yes"
            #         )

            env = Environment(
                    experiment_name=f'play_winner_ES_enemy{en}',
                    enemies=[en],
>>>>>>> 7d86e11d2ce18f46e7f6f665848a748f2cd9e641
                    playermode = 'ai',
                    enemymode='static',
                    sound='off',
                    player_controller=player_controller(n_hidden_neurons),
                    level=2,
                    speed='fastest',
                    logs='off',
                    randomini="yes"
                    )            
            
    
<<<<<<< HEAD
        for i in range(1,11,1):
            print(f'Experiment {i}:')
=======
            # for i in range(1,11,1):
            #     print(f'enemy {en}, experiment {i}:')
>>>>>>> 7d86e11d2ce18f46e7f6f665848a748f2cd9e641
                
            #     genome_SGA = np.load(f'results_ES/{name}/overall_best.npy')
            #     print(genome_SGA)
            #     mean_SGA = play_winner(env, genome_SGA)
            #     means_SGA.append(mean_SGA)


            for i in range(1,11,1):
                print(f'enemy {en}, experiment {i}:')
                
<<<<<<< HEAD
            genome_SGA = np.load(f'results_SGA2/{name}/overall_best.npy')
            mean_SGA = play_winner_SGA(SGA_env, genome_SGA)
            means_SGA.append(mean_SGA)
        print(means_NEAT)
        print(means_SGA)
        plot_boxplot(means_SGA)
            
            #np.save(f'results_NEAT/final_experiment/enemy{en}_mean_gains.npy', means_NEAT)
        np.save(f'results_SGA2/{name}/e{en}_mean_gains.npy', means_SGA)
=======
                genome_ES = np.load(f'results_ES/{name}/overall_best.npy')
                # print(genome_ES)
                mean_ES = play_winner(env, genome_ES)
                means_ES.append(mean_ES)
            # print(means_ES)
            # print(means_SGA)
            # plot_boxplot(means_SGA, en)
            plot_boxplot(means_ES, en)
            
            #np.save(f'results_NEAT/final_experiment/enemy{en}_mean_gains.npy', means_NEAT)
            np.save(f'results_ES/{name}/e{en}_mean_gains.npy', means_SGA)
>>>>>>> 7d86e11d2ce18f46e7f6f665848a748f2cd9e641
