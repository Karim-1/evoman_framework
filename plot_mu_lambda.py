import csv
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm

fitnesses = []
with open('results_es/mu_lambda_experiment/max_fitnesses.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        print(row)
        fitnesses.append(row)


fig = plt.figure()
ax = fig.gca(projection='3d')

MUs= range(25,101,25)
LAMBDAs = range(25,101,25)
X, Y = np.meshgrid(MUs, LAMBDAs)
Z = np.array([list( map(float,i) ) for i in fitnesses])

surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False)

ax.set_xlabel('Mu')
ax.set_ylabel('Lambda')
ax.set_zlabel('Fitness')
plt.show()