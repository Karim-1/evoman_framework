from plot_functions import plot_fitness, plot_best

import matplotlib.pyplot as plt
import numpy as np

# experiment_name = 'unconnected_results'

# mean_fitness = np.load(f'results/{experiment_name}/mean_fitness.npy')
# best_fitness = np.load(f'results/{experiment_name}/best_fitness.npy')

# print(best_fitness)

# plot_fitness(mean_fitness, best_fitness)

def plot_pop_experiment():
    popsizes = [50,150]
    colors = ['red','blue']
    enemies = [2,4,5]
    generations = range(25)

    mean_max_fitness = []

    best_50_en2 = np.load(f'results/undirected_pop50_en2/best_fitness.npy')
    best_50_en4 = np.load(f'results/undirected_pop50_en4/best_fitness.npy')
    best_50_en5 = np.load(f'results/undirected_pop50_en5/best_fitness.npy')
    best_50_avg = [(x+y+z) / 3 for x, y, z in zip(best_50_en2, best_50_en4, best_50_en5)]

    best_150_en2 = np.load(f'results/undirected_pop150_en2/best_fitness.npy')
    best_150_en4 = np.load(f'results/undirected_pop150_en4/best_fitness.npy')
    best_150_en5 = np.load(f'results/undirected_pop150_en5/best_fitness.npy')
    best_150_avg = [(x+y+z) / 3 for x, y, z in zip(best_150_en2, best_150_en4, best_150_en5)]
    
    
    plt.plot(generations, best_50_avg, color = 'red', label = '50')
    plt.plot(generations, best_150_avg, color = 'blue', label = '150')

    plt.ylabel('fitness')
    plt.xlabel('generations')
    plt.ylim(0,100)
    plt.grid()
    plt.legend(title='Population size:', loc='lower right')
    plt.show()

plot_pop_experiment()