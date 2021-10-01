import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# use big font size so that it is readable in latex
matplotlib.rcParams.update({'font.size': 12})

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


def plot_final_experiment():
    generations = range(15)
    enemies = [2,4,5]
    enemynames = ['Airman (enemy 2)', 'HeatMan (enemy 4)', 'Metalman (enemy 5)']
    colors = ['_', '_', 'blue', '_', 'red', 'black']
    simulations = [0,1,2,4,5,6,7,8,9]
    
    best_fitness = {}
    mean_fitness = {}
    
    best_average_fitness = {}
    mean_average_fitness = {}

    # add all mean and best fitnesses to lists in the dictionary
    for en in enemies:
        best_fitness[en] = []
        mean_fitness[en] = []

        for i in simulations:
            best_fitness[en].append(np.load(f'results/final_enemy{en}_{i}/best_fitness.npy'))
            mean_fitness[en].append(np.load(f'results/final_enemy{en}_{i}/mean_fitness.npy'))
            # print(f'enemy {en}', max(np.load(f'results/final_enemy{en}_{i}/best_fitness.npy')))

    fig, axs = plt.subplots(3,1, figsize=(5,6.5))
    i = 0

    # retrieve average of the 10 runs and plot with confidence interval
    for en in enemies:
        best_average_fitness[en] = [np.mean(k) for k in zip(*best_fitness[en])]
        mean_average_fitness[en] = [np.mean(k) for k in zip(*mean_fitness[en])]
    
        ci = np.std(best_average_fitness[en])
        axs[i].plot(generations, best_average_fitness[en], color = colors[en])
        axs[i].fill_between(generations, (best_average_fitness[en]-ci), (best_average_fitness[en]+ci), color=colors[en], alpha=.1)
        axs[i].set_ylim(0,100)
        axs[i].set_ylabel('fitness')

        ci = np.std(mean_average_fitness[en])
        axs[i].plot(generations, mean_average_fitness[en], color = colors[en], label=f'enemy {en}', linestyle = 'dashed')
        axs[i].fill_between(generations, mean_average_fitness[en]-ci, mean_average_fitness[en]+ci, color=colors[en], alpha=.1)
        axs[i].grid()
        axs[i].set_title(f'{enemynames[i]}', fontsize=12,color=colors[en])
        
        i+=1
        
    
    
    plt.ylim(0,100)
    plt.xlabel('Generation')
    # plt.title("Average of mean and best fitness for each enemy in 10 runs")
    # plt.legend()
    # plt.grid()
    plt.tight_layout()
    plt.savefig(f'plots/final_experiments_neat')
    plt.show()


if __name__ == "__main__":
    # plot_pop_experiment()
    plot_final_experiment()