# SGA and NEAT algorithm to train an AI for the EVOMAN framework

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages.

```bash
pip install requirements.txt
```

## file usage

### SGA

run `SGA.py` to run 1 experiment, the 10 experiments have to be run manually, and thus also the enemy and experiment name should be changed accordingly.

### NEAT

run `neat_algo.py` to run the 10 experiments

## results

if you want to play with a winning genome of the NEAT algorithm and of the SGA, run `play_winner.py`. Make sure to change to folder to the genome you want to use.

The numpy lists with the mean and max fitness values can be found in `results_SGA` and `results_NEAT`.

In the `results_NEAT` folder, the main results of the 10 simulations can be found in `final_experiment`, furthermore, the results of the different population sizes are in `population_experiment` and the experiment with different initial connections between genome nodes are in `initial_conditions`.

In the `results_SGA` folder, the main results of the 10 simulations can be found, furthermore, in the folder `Parameter experiments`, the results of the experiments with the population size and the number of hidden nodes can be found.

in `plot_functions.py` different plot functions can be found that were used for the results in the paper

In the `plots` folder, the plots used in the report can be found.
