import matplotlib.pyplot as plt

def plot_fitness(mean_fitness, best_fitness, name):
    fig, axs = plt.subplots(2)

    axs[0].plot(range(1, len(best_fitness)+1), best_fitness, '-o', label='Best genome fitness')
    axs[0].set_ylabel('Best fitness')
    axs[1].plot(range(1, len(mean_fitness)+1), mean_fitness, '-o', label='Mean genome fitness')
    axs[1].set_xlabel('Generation')
    axs[1].set_ylabel('Mean fitness')
    
    # plt.legend()
    plt.savefig(name+'.png')
    plt.show()