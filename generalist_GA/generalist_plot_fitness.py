import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Directory where the experiment results are stored
experiment_name = 'generalist_ga_experiment'

# Function to load fitness history from a run
def load_fitness_history(run_id):
    run_dir = os.path.join(experiment_name, f'run_{run_id}')
    fitness_file = os.path.join(run_dir, 'fitness_history.json')
    
    if os.path.exists(fitness_file):
        with open(fitness_file, 'r') as f:
            fitness_data = json.load(f)
        return fitness_data['mean_fitness_per_gen'], fitness_data['best_fitness_per_gen']
    else:
        raise FileNotFoundError(f"Fitness history not found for run {run_id}")

# Function to calculate average and standard deviation of fitness over runs
def calculate_avg_std(group_runs):
    mean_fitness_all_runs = []
    max_fitness_all_runs = []
    
    for run_id in group_runs:
        mean_fitness, max_fitness = load_fitness_history(run_id)
        mean_fitness_all_runs.append(mean_fitness)
        max_fitness_all_runs.append(max_fitness)
    
    # Convert to numpy arrays for easy computation
    mean_fitness_all_runs = np.array(mean_fitness_all_runs)
    max_fitness_all_runs = np.array(max_fitness_all_runs)
    
    # Calculate mean and standard deviation
    mean_fitness_avg = np.mean(mean_fitness_all_runs, axis=0)
    mean_fitness_std = np.std(mean_fitness_all_runs, axis=0)
    
    max_fitness_avg = np.mean(max_fitness_all_runs, axis=0)
    max_fitness_std = np.std(max_fitness_all_runs, axis=0)
    
    return mean_fitness_avg, mean_fitness_std, max_fitness_avg, max_fitness_std

# Function to plot fitness data and save to file
def plot_fitness_data(mean_avg, mean_std, max_avg, max_std, group_name, save_path=None):
    generations = np.arange(1, len(mean_avg) + 1)
    
    plt.figure(figsize=(10, 6))
    
    # Plot mean fitness
    plt.plot(generations, mean_avg, label='Mean Fitness (Avg)', color='blue')
    plt.fill_between(generations, mean_avg - mean_std, mean_avg + mean_std, color='blue', alpha=0.2)
    
    # Plot max fitness
    plt.plot(generations, max_avg, label='Max Fitness (Avg)', color='green')
    plt.fill_between(generations, max_avg - max_std, max_avg + max_std, color='green', alpha=0.2)
    
    # Labels and title
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'Fitness over Generations - {group_name}')
    plt.legend()
    plt.grid(True)
    
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    # Display the plot
    plt.show()

# Group 1: run_1 to run_10
group_1_runs = range(1, 11)
mean_avg_1, mean_std_1, max_avg_1, max_std_1 = calculate_avg_std(group_1_runs)
plot_fitness_data(mean_avg_1, mean_std_1, max_avg_1, max_std_1, "Group 1 (run_1 to run_10)", 
                  save_path="group1_fitness_plot.png")

# Group 2: run_11 to run_20
group_2_runs = range(11, 21)
mean_avg_2, mean_std_2, max_avg_2, max_std_2 = calculate_avg_std(group_2_runs)
plot_fitness_data(mean_avg_2, mean_std_2, max_avg_2, max_std_2, "Group 2 (run_11 to run_20)", 
                  save_path="group2_fitness_plot.png")
