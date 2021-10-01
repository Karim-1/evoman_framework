import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# use big font size so that it is readable in latex
matplotlib.rcParams.update({'font.size': 14})

def plot_fitness(mean_fitness, best_fitness, experiment_name):
    generations = range(len(best_fitness))
    
    plt.errorbar(generations, mean_fitness, yerr=np.std(mean_fitness), label='mean')
    plt.errorbar(generations, best_fitness, yerr=np.std(best_fitness), label='best')
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.title("Population's average and best fitness")
    plt.grid()
    plt.tightlayout()
    plt.legend()
    plt.savefig(f'plots/{experiment_name}_fitness_plot')
    plt.show()



def plot_pop_experiment():
    generations = range(25)

    best_50_en2 = np.load(f'results/population_experiment/undirected_pop50_en2/best_fitness.npy')
    best_50_en4 = np.load(f'results/population_experiment/undirected_pop100_en4/best_fitness.npy')
    best_50_en5 = np.load(f'results/population_experiment/undirected_pop50_en5/best_fitness.npy')
    best_50_avg = [(x+y+z) / 3 for x, y, z in zip(best_50_en2, best_50_en4, best_50_en5)]

    best_100_en2 = np.load(f'results/population_experiment/undirected_pop100_en2/best_fitness.npy')
    best_100_en4 = np.load(f'results/population_experiment/undirected_pop100_en4/best_fitness.npy')
    best_100_en5 = np.load(f'results/population_experiment/undirected_pop100_en5/best_fitness.npy')
    best_100_avg = [(x+y+z) / 3 for x, y, z in zip(best_100_en2, best_100_en4, best_100_en5)]

    best_150_en2 = np.load(f'results/population_experiment/undirected_pop150_en2/best_fitness.npy')
    best_150_en4 = np.load(f'results/population_experiment/undirected_pop100_en4/best_fitness.npy')
    best_150_en5 = np.load(f'results/population_experiment/undirected_pop150_en5/best_fitness.npy')
    best_150_avg = [(x+y+z) / 3 for x, y, z in zip(best_150_en2, best_150_en4, best_150_en5)]
    
    
    plt.plot(generations, best_50_avg, color = 'red', label = '50')
    plt.plot(generations, best_100_avg, color = 'blue', label = '100')
    plt.plot(generations, best_150_avg, color = 'black', label = '150')

    plt.ylabel('fitness')
    plt.xlabel('generations')
    plt.ylim(0,80)
    plt.grid()
    plt.tight_layout()
    plt.legend(title='Population size:', loc='lower right')
    plt.show()


def plot_best(tba):
    pass

if __name__ == "__main__":
    plot_pop_experiment()