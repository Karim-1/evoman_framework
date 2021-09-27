import matplotlib.pyplot as plt

def plot_fitness(mean_fitness, best_fitness):
    plt.plot(range(1, len(mean_fitness)+1), '-o', label='Mean genome fitness')
    plt.plot(range(1, len(best_fitness)+1), '-o', label='Best genome fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()