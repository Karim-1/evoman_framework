import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

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
    plt.tight_layout()
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

def plot_ES():
    generations = range(21)

    path = 'results_ES'
    with open(f'{path}/en[2,5,6]/log.pkl', 'rb') as file:
        log = pickle.load(file)
    # log = pickle.load(f'{path}/en[2,5,6]/log.pkl')


    max_fits256 = []
    avg_fits256 = []
    std_fits256 = []

    for fit in log.select('max'):
        max_fits256.append(fit[0])
    for fit in log.select('avg'):
        avg_fits256.append(fit[0])
    for fit in log.select('std'):
        std_fits256.append(fit[0])
    
    mean_std_256 = np.mean(std_fits256)
    plt.plot(generations, max_fits256, label='max fitness [2,5,6]')
    plt.fill_between(generations, max_fits256-mean_std_256, max_fits256+mean_std_256, alpha=.1)
    plt.show()
    
    

def plot_final_experiment():
    generations = range(20)
    generations_SGA = range(20)
    enemies = [1, 2]
    en_dict = {1: [2,4,5], 2: [7,8]}
    enemynames = ['Airman (enemy 2)', 'HeatMan (enemy 4)', 'Metalman (enemy 5)']
    colors = ['_', '_', 'blue', '_', 'red', 'black']
    colors_SGA = ['_', '_', 'green', '_', 'orange', 'brown']
    simulations = range(10)
    
    best_fitness = {}
    mean_fitness = {}
    best_fitness_SGA = {}
    mean_fitness_SGA = {}
    
    best_average_fitness = {}
    mean_average_fitness = {}
    best_average_fitness_SGA = {}
    mean_average_fitness_SGA = {}

    # add all mean and best fitnesses to lists in the dictionary
    for en in enemies:
        best_fitness[en] = []
        mean_fitness[en] = []
        best_fitness_SGA[en] = []
        mean_fitness_SGA[en] = []

        for i in simulations:
            best_fitness[en].append(np.load(f'results_NEAT/final_experiment/final_enemy{en}_{i}/best_fitness.npy'))
            mean_fitness[en].append(np.load(f'results_NEAT/final_experiment//final_enemy{en}_{i}/mean_fitness.npy'))
            best_fitness_SGA[en].append(np.load(f'results_SGA2/en{en_dict[en]}-{i+1}/best_fitness.npy'))
            mean_fitness_SGA[en].append(np.load(f'results_SGA2/en{en_dict[en]}-{i+1}/mean_fitness.npy'))

    fig, axs = plt.subplots(3,1, figsize=(6,7))
    i = 0

    # retrieve average of the 10 runs for both NEAT and SGA and plot with confidence interval
    for en in enemies:
        best_average_fitness[en] = [np.mean(k) for k in zip(*best_fitness[en])]
        mean_average_fitness[en] = [np.mean(k) for k in zip(*mean_fitness[en])]
        best_average_fitness_SGA[en] = [np.mean(k) for k in zip(*best_fitness_SGA[en])]
        mean_average_fitness_SGA[en] = [np.mean(k) for k in zip(*mean_fitness_SGA[en])]
        
        ci = np.std(best_average_fitness[en])
        ci_SGA = np.std(best_average_fitness_SGA[en])
        axs[i].plot(generations, best_average_fitness[en], color = colors[en], label = 'Max NEAT')
        axs[i].plot(generations_SGA, best_average_fitness_SGA[en], color = colors_SGA[en], label = 'Max SGA')
        axs[i].fill_between(generations, (best_average_fitness[en]-ci), (best_average_fitness[en]+ci), color=colors[en], alpha=.1)
        axs[i].fill_between(generations_SGA, (best_average_fitness_SGA[en]-ci_SGA), (best_average_fitness_SGA[en]+ci_SGA), color=colors_SGA[en], alpha=.1)
        axs[i].set_ylim(0,100)
        axs[i].set_ylabel('fitness', fontsize=16)

        ci = np.std(mean_average_fitness[en])
        ci_SGA = np.std(mean_average_fitness_SGA[en])
        axs[i].plot(generations, mean_average_fitness[en], color = colors[en], linestyle = 'dashed', label = 'Mean NEAT')
        axs[i].plot(generations_SGA, mean_average_fitness_SGA[en], color = colors_SGA[en], linestyle = 'dashed', label = 'MEAN SGA')
        axs[i].fill_between(generations, mean_average_fitness[en]-ci, mean_average_fitness[en]+ci, color=colors[en], alpha=.1)
        axs[i].fill_between(generations_SGA, mean_average_fitness_SGA[en]-ci_SGA, mean_average_fitness_SGA[en]+ci_SGA, color=colors_SGA[en], alpha=.1)
        axs[i].grid()
        axs[i].set_title(f'{enemynames[i]}', fontsize=16, color=colors[en])
        axs[i].legend(loc="lower right", bbox_to_anchor=(1.04,-0.1))
              
        i+=1
    
    
    plt.ylim(0,100)
    plt.xlabel('Generation', fontsize=16)
    # plt.title("Average of mean and best fitness for each enemy in 10 runs")
    # plt.legend()
    # plt.grid()
    plt.tight_layout()
    plt.savefig(f'plots/final_experiments_neat+SGA')
    plt.show()
    
def plot_boxplot(data_NEAT, data_SGA, enemy):
    enemynames = ['_', '_', 'Airman (enemy 2)', '_', 'HeatMan (enemy 4)', 'Metalman (enemy 5)']
    colors = ['_', '_', 'blue', '_', 'red', 'black']
    #Make a boxplot of two arrays of data             
    data = [data_NEAT, data_SGA]
    
    fig, ax = plt.subplots(figsize =(6, 6))
    ax.set_title(f'{enemynames[enemy]}', fontsize=18, color=colors[enemy])
    ax.boxplot(data)
    plt.ylabel('individual gain', fontsize=18)
    ax.set_xticklabels(['NEAT', 'SGA'], fontsize=18)
    plt.savefig(f'plots/boxplot_enemy{enemy}')
    plt.show()


if __name__ == "__main__":
    # plot_pop_experiment()
    # plot_final_experiment()
    plot_ES()