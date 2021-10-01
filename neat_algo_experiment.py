import sys, os

# import classes
sys.path.insert(0, 'evoman') 
from demo_controller import NEAT_controller
from environment import Environment
from neat_algo import run_neat


experiment_name = 'final'
enemies = [2,4,5]

for en in enemies:
    for i in range(10):
        if not os.path.exists(f'results/{experiment_name}{i}'):
            os.makedirs(f'results/{experiment_name}{i}')
        
        # retrieve configuration file
        local_dir = os.path.dirname(__file__)
        config_file = os.path.join(local_dir, "neat_config.txt")

        # remove game display
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        
        # initialize environment with NEAT network as player controller
        env = Environment(
            experiment_name=experiment_name,
            enemies=[en],
            playermode = 'ai',
            enemymode='static',
            sound='off',
            player_controller=NEAT_controller(),
            level=2,
            speed='fastest',
            logs='off'
            )

        # lists to store information for each generation
        mean_fitness = []
        best_fitness = []

        # run experiments
        run_neat(config_file, experiment_name, env)