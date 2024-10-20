import os
import numpy as np
import matplotlib.pyplot as plt
from evoman.environment import Environment
from demo_controller import player_controller

# Set number of hidden neurons for the neural network
n_hidden_neurons = 10

# Directory where results are stored
experiment_name = 'generalist_ga_experiment'

# Function to test a set of weights against all enemies and calculate Gain
def test_weights_and_calculate_gain(weights):
    enemies = [1, 2, 3, 4, 5, 6, 7, 8]  # Test against all enemies
    gains = []

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
        
        # Calculate Gain (new formula)
        individual_gain = player_life - enemy_life
        gains.append(individual_gain)
        print(f"Tested against enemy {enemy}: Player life = {player_life}, Enemy life = {enemy_life}, Gain = {individual_gain}")
    
    return gains

# Function to load best weights from a file
def load_best_weights(file_path):
    return np.loadtxt(file_path)

# Function to test best weights for a group and return gains for each run
def get_gains_for_group(group_runs):
    all_gains = []

    for run_id in group_runs:
        run_dir = os.path.join(experiment_name, f'run_{run_id}')
        weights_file = os.path.join(run_dir, 'best_weights.txt')

        if os.path.exists(weights_file):
            print(f"Testing best weights from run {run_id}...")
            best_weights = load_best_weights(weights_file)
            
            # Test the best weights against all enemies and calculate gains
            gains = test_weights_and_calculate_gain(best_weights)
            
            # Sum of gains for all enemies in this run
            total_gain = np.sum(gains)
            all_gains.append(total_gain)
        else:
            print(f"Best weights file not found for run {run_id}")
    
    return all_gains

# Function to plot and save a box plot of gains for a group
def plot_gains_boxplot(gains, group_name, save_path):
    plt.figure(figsize=(8, 6))
    
    # Create box plot
    plt.boxplot(gains, labels=[group_name])
    
    # Labels and title
    plt.ylabel('Total Gain')
    plt.title(f'Total Gain for Each Run in {group_name}')
    plt.grid(True)
    
    # Save the plot to the specified path
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
    # Show plot
    plt.show()

# Group 1: run_1 to run_10
group_1_runs = range(1, 11)
group_1_gains = get_gains_for_group(group_1_runs)
plot_gains_boxplot(group_1_gains, "Group 1 (run_1 to run_10)", save_path="group1_gains_boxplot.png")

# Group 2: run_11 to run_20
group_2_runs = range(11, 21)
group_2_gains = get_gains_for_group(group_2_runs)
plot_gains_boxplot(group_2_gains, "Group 2 (run_11 to run_20)", save_path="group2_gains_boxplot.png")
