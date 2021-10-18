import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({'font.size': 18})

gradual_avg = []; gradual_max = []; gradual_std = []

static_avg = []; static_max = []; static_std = []

for i in range(10):
    gradual = np.load(f'results_ES/fmutpb_experiment_gradual{i}.npz')
    gradual_avg.append(gradual['name1'])
    gradual_max.append(gradual['name2'])
    gradual_std.append(gradual['name3'])

    static = np.load(f'results_ES/fmutbp_experiment_static{i}.npz')
    
    static_avg.append(static['name1'])
    static_max.append(static['name2'])
    static_std.append(static['name3'])

    

mean_gradual_avg = [np.mean(k) for k in zip(*gradual_avg)]
mean_gradual_max = [np.mean(k) for k in zip(*gradual_max)]
mean_gradual_std = [np.mean(k) for k in zip(*gradual_std)]

mean_static_avg = [np.mean(k) for k in zip(*static_avg)]
mean_static_max = [np.mean(k) for k in zip(*static_max)]
mean_static_std = [np.mean(k) for k in zip(*static_std)]

gens = range(21)

# plt.plot(gens, mean_gradual_max, label ='gradual')
# plt.plot(gens, mean_static_max, label = 'static')
# plt.fill_between(gens, (mean_gradual_max - np.mean(mean_gradual_std)), (mean_gradual_max + np.mean(mean_gradual_std)), alpha=.1)
# plt.fill_between(gens, (mean_static_max- np.mean(mean_static_std)), (mean_static_max + np.mean(mean_static_std)), alpha=.1)



plt.plot(gens, mean_gradual_avg, label='exponential decrease')
plt.plot(gens, mean_static_avg, label='static')
plt.fill_between(gens, (mean_gradual_avg - np.mean(mean_gradual_std)), (mean_gradual_avg + np.mean(mean_gradual_std)), alpha=.1)
plt.fill_between(gens, (mean_static_avg- np.mean(mean_static_std)), (mean_static_avg+ np.mean(mean_static_std)), alpha=.1)

plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.tight_layout()
plt.legend(loc='center right')

plt.show()

