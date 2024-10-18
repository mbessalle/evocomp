import numpy as np
import os
from evoman.environment import Environment
from demo_controller import player_controller

# Number of hidden neurons in the neural network
n_hidden_neurons = 10

# Initialize the environment for testing
experiment_name = 'generalist_ga_experiment'

# Function to test a set of weights against all enemies and count victories
def test_weights_against_all_enemies(weights):
    # List of enemies to test
    enemies = [1, 2, 3, 4, 5, 6, 7, 8]
    victories = 0

    for enemy in enemies:
        # Initialize the environment for each enemy
        env = Environment(experiment_name=experiment_name,
                          enemies=[enemy],  # Test one enemy at a time
                          multiplemode="no",
                          playermode="ai",
                          player_controller=player_controller(n_hidden_neurons),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          visuals=False)
        
        # Play the game with the given weights
        fitness, player_life, enemy_life, time = env.play(pcont=weights)
        
        # Check if the player won (enemy defeated)
        if enemy_life <= 0 and player_life > 0:
            victories += 1
        print(f"Tested against enemy {enemy}: Player life = {player_life}, Enemy life = {enemy_life}")
    
    return victories

# Function to load best weights from a file
def load_best_weights(file_path):
    return np.loadtxt(file_path)

# Function to test all the best weights from each run
def test_all_best_weights():
    # Directory where results for each run are stored
    for run_id in range(1, 11):
        run_dir = os.path.join(experiment_name, f'run_{run_id}')
        weights_file = os.path.join(run_dir, 'best_weights.txt')

        if os.path.exists(weights_file):
            print(f"Testing best weights from run {run_id}...")
            best_weights = load_best_weights(weights_file)
            
            # Test the best weights against all enemies
            victories = test_weights_against_all_enemies(best_weights)
            print(f"Run {run_id}: Defeated {victories} enemies\n")
        else:
            print(f"Best weights file not found for run {run_id}")

# Execute the test for all runs
test_all_best_weights()
