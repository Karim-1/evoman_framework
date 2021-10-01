# SGA and NEAT algorithm to train an AI for the EVOMAN framework

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages.

```bash
pip install requirements.txt
```

## file usaga

run `neat_algo` to run the 10 experiments

in `plot_functions` different plot functions can be found that were used for the results in the paper

## results

The numpy lists with the mean and max fitness values can be found in `results_SGA` and `results_NEAT`

In the `results_NEAT` folder, the main results of the 10 simulations can be found in `final_experiment`, furthermore, the results of the different population sizes are in `population_experiment` and the experiment with different initial connections between genome nodes are in `initial_conditions`.
