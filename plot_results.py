from plot_functions import plot_fitness, plot_best

import matplotlib.pyplot as plt
import numpy as np

experiment_name = 'unconnected_results'

mean_fitness = np.load(f'results/{experiment_name}/mean_fitness.npy')
best_fitness = np.load(f'results/{experiment_name}/best_fitness.npy')

print(best_fitness)

# plot_fitness(mean_fitness, best_fitness)

def plot_pop_experiment():
    popsizes = [50,100,150,200]
    colors = ['red','green', 'blue']
    enemies = [2,4,5]
    generations = range(10)

    
    for i in range(len(enemies)):
        for popsize in popsizes:
            plt.plot(generations, np.load('undirected_pop{popsize}_en{enemies[i]}'), col = colors[i], label = f'Enemy {enemies[i]}')

plt.ylim(0,100)
plt.grid()
plt.legend()
plt.show()
