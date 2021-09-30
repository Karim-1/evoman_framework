from plot_functions import plot_fitness, plot_best

import matplotlib.pyplot as plt
import numpy as np

experiment_name = 'unconnected_results'

mean_fitness = np.load(f'results/{experiment_name}/mean_fitness.npy')
best_fitness = np.load(f'results/{experiment_name}/best_fitness.npy')

print(best_fitness)

# plot_fitness(mean_fitness, best_fitness)