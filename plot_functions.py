import matplotlib.pyplot as plt
import numpy as np

def plot_fitness(mean_fitness, best_fitness, experiment_name):
    generations = range(len(best_fitness))
    
    plt.errorbar(generations, mean_fitness, yerr=np.std(mean_fitness), label='mean')
    plt.errorbar(generations, best_fitness, yerr=np.std(best_fitness), label='best')
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.title("Population's average and best fitness")
    plt.grid()
    plt.legend()
    plt.savefig(f'plots/{experiment_name}_fitness_plot')
    plt.show()

def plot_best(tba):
    pass