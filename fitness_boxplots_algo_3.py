import numpy as np
import os
import matplotlib.pyplot as plt
from evoman.environment import Environment
from demo_controller import player_controller

experiment_name = "weight_test"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10
enemies = [1, 2, 3]
tries = 10

env = Environment(
    experiment_name=experiment_name,
    enemies=enemies,
    multiplemode="yes",
    playermode="ai",
    player_controller=player_controller(n_hidden_neurons),
    speed="fastest",
    enemymode="static",
    level=2,
    visuals=False,
)

def simulation(env, x, min=False):
    f, p, e, t = env.play(pcont=x)
    if not min:
        return f
    else:
        f = (f - (-100)) / (100 - (-100)) * 100
        return abs(f - 100)

def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env, y), x)))

# Dictionary to store individual gains for each enemy
individual_gains = {enemy: [] for enemy in enemies}

for attempt in range(1, tries + 1):
    for enemy in enemies:
        # Load population weights
        with open(f"results/population_level{enemy}_run{attempt}") as f:
            pop = f.readlines()
        
        for i in range(len(pop)):
            pop[i] = pop[i].split(",")
            pop[i] = [float(x.strip()) for x in pop[i]]

        # Load fitness values
        with open(f"results/fitness_level{enemy}_run{attempt}") as f:
            fit = f.readlines()
        
        for i in range(len(fit)):
            fit[i] = fit[i].split(",")
            fit[i] = [float(x.strip()) for x in fit[i]]

        # Find the best individual (maximum fitness)
        max_i = np.argmax(fit)
        best_individual = pop[max_i]
        individual_gain = 0

        # Simulate the best individual 5 times to calculate the average individual gain
        for i in range(5):
            f, p, e, t = env.play(pcont=np.array(best_individual))
            individual_gain += p - e

        # Store the average gain for this run in the dictionary
        individual_gains[enemy].append(individual_gain / 5)

        # Save individual gain for each run (optional, if needed for record-keeping)
        np.savetxt(
            f"results/individual_gain_level{enemy}_run{attempt}",
            [individual_gain / 5],
            delimiter=","
        )

# Create a list of gains for each enemy to plot
gains_to_plot = [individual_gains[enemy] for enemy in enemies]

# Create the boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(gains_to_plot)

# Set the x-axis labels to the enemies
plt.xticks([1, 2, 3], [f'Enemy {enemy}' for enemy in enemies])

# Label the axes
plt.xlabel("Enemies", fontsize=12)
plt.ylabel("Individual Gain", fontsize=12)

# Set y-axis limits from -100 to 100
plt.ylim(-100, 100)

# Title of the plot
plt.title("Boxplot of Individual Gain for Different Enemies", fontsize=14)

#Save the boxplot
plt.savefig("plots/individual_gain_boxplot_algo3.png")

# Show the plot
plt.show()