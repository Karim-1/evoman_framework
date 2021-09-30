from neat_algo import run_neat
import sys, os
    
for i in range(5,10,1):
    experiment_name = 'neat'
    enemies = [2]

    if not os.path.exists(f'results/{experiment_name}_results'):
        os.makedirs(f'results/{experiment_name}_results')
    
    # retrieve configuration file
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat_config.txt")

    # remove game display
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    # initialize environment with NEAT network as player controller
    env = Environment(
        experiment_name=experiment_name,
        enemies=enemies,
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
    run_neat(config_file, experiment_name)