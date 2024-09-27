#######################################################################################
# EvoMan Framework - Genetic Algorithm for Specialist Controller                      #
# Author: Group 19                                                                    #
# Assignment: Task I - Specialist Agent Training using GA                             #
#######################################################################################

# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
from evoman.environment import Environment
from demo_controller import player_controller

# choose this for not using visuals and thus making experiments faster (comment this out and set visuals to True in env to watch the game training)
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Experiment parameters
experiment_name = 'ga_specialist_task'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Initialize environment parameters (use enemy 1, 2, 3 separately in experiments)
n_hidden_neurons = 10
enemies = [1, 2, 3]  # Use 3 different enemies in your experiments
runs_per_experiment = 10

# Initialize the environment before calculating n_vars
enemy = 1  # Example for enemy 1, this can be changed for each experiment
env = Environment(experiment_name=experiment_name,
                  enemies=[enemy],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# Number of weights for the perceptron with n_hidden_neurons
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

# Genetic Algorithm parameters
npop = 100  # Population size
gens = 30   # Number of generations
mutation_rate = 0.2
dom_u = 1
dom_l = -1
notimproved_limit = 15


def simulation(env, individual):
    """Runs the simulation and returns the fitness and energy details."""
    fitness, player_energy, enemy_energy, time = env.play(pcont=individual)
    return fitness, player_energy, enemy_energy


def normalize_fitness(fitness, fitness_pop):
    min_fit, max_fit = min(fitness_pop), max(fitness_pop)
    return (fitness - min_fit) / (max_fit - min_fit) if max_fit != min_fit else 0.0001


def tournament_selection(pop, fitness_pop):
    selected = np.random.choice(len(pop), 2)
    return pop[selected[0]] if fitness_pop[selected[0]] > fitness_pop[selected[1]] else pop[selected[1]]


def crossover_and_mutate(pop, fitness_pop):
    offspring = np.zeros((npop, n_vars))
    for i in range(0, npop, 2):
        parent1 = tournament_selection(pop, fitness_pop)
        parent2 = tournament_selection(pop, fitness_pop)
        cross_point = np.random.randint(1, n_vars)
        offspring[i, :cross_point] = parent1[:cross_point]
        offspring[i, cross_point:] = parent2[cross_point:]
        offspring[i+1, :cross_point] = parent2[:cross_point]
        offspring[i+1, cross_point:] = parent1[cross_point:]
        # Mutation step
        for child in [offspring[i], offspring[i+1]]:
            if np.random.rand() < mutation_rate:
                child += np.random.normal(0, 1, n_vars)
    return np.clip(offspring, dom_l, dom_u)


def run_ga_experiment(env, enemy, runs_per_experiment, gens):
    """Runs GA experiment for a specific enemy."""
    results = []
    best_solutions = []  # To store the best solution (weights) for each run

    for run in range(runs_per_experiment):
        population = np.random.uniform(dom_l, dom_u, (npop, n_vars))
        fitness_pop = np.array([simulation(env, individual)[0] for individual in population])
        best_fitness = max(fitness_pop)
        generation_results = []

        for gen in range(gens):
            offspring = crossover_and_mutate(population, fitness_pop)
            offspring_fitness = np.array([simulation(env, child)[0] for child in offspring])
            
            # Merge offspring and select survivors
            population = np.vstack((population, offspring))
            fitness_pop = np.hstack((fitness_pop, offspring_fitness))
            survivors_idx = np.argsort(fitness_pop)[-npop:]
            population = population[survivors_idx]
            fitness_pop = fitness_pop[survivors_idx]
            
            # Record generation best and average
            best_idx = np.argmax(fitness_pop)
            best_fitness = fitness_pop[best_idx]
            mean_fitness = np.mean(fitness_pop)
            generation_results.append((best_fitness, mean_fitness))

            # Early stopping
            if gen - np.argmax(fitness_pop) > notimproved_limit:
                break

        # Store the best solution from this run
        best_solutions.append(population[best_idx])

        # Save results of this run
        results.append(generation_results)
    
    return results, best_solutions

# Visualization
def plot_results(results, enemy):
    """Plot results for fitness (mean and max) over generations."""
    
    # Find the minimum number of generations across all runs
    min_gens = min([len(run) for run in results])

    # Adjust the results to truncate to the minimum number of generations
    truncated_results = [run[:min_gens] for run in results]

    # Calculate mean fitnesses for truncated results
    mean_fitnesses = np.mean([[gen[1] for gen in run] for run in truncated_results], axis=0)
    max_fitnesses = np.mean([[gen[0] for gen in run] for run in truncated_results], axis=0)
    
    # Plotting
    plt.figure()
    plt.plot(range(min_gens), mean_fitnesses, label='Mean Fitness', color='blue')
    plt.plot(range(min_gens), max_fitnesses, label='Max Fitness', color='red')
    plt.title(f'GA Fitness Over Generations - Enemy {enemy}')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.savefig(f'{experiment_name}/fitness_plot_enemy_{enemy}.png')

# Run experiments for each enemy
if __name__ == "__main__":
    for enemy in enemies:
        # Set up the environment for the current enemy
        env = Environment(experiment_name=experiment_name,
                          enemies=[enemy],
                          playermode="ai",
                          player_controller=player_controller(n_hidden_neurons),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          visuals=False)
        
        # Run GA experiment for the enemy
        print(f"Running GA experiment for enemy {enemy}...")
        results, best_solutions = run_ga_experiment(env, enemy, runs_per_experiment, gens)
        
        # Plot results
        plot_results(results, enemy)
        
        # Save and print best solutions and gains
        for run_idx, best_solution in enumerate(best_solutions):
            # Save best solution
            np.savetxt(f'{experiment_name}/best_solution_enemy_{enemy}_run_{run_idx}.txt', best_solution)
            
            # Re-run simulation with best solution to get the individual gain
            fitness, player_energy, enemy_energy = simulation(env, best_solution)
            individual_gain = player_energy - enemy_energy
            
            # Print hyperparameters, weights, and gain
            print(f"\nRun {run_idx + 1} for Enemy {enemy}:")
            print(f"Best Fitness: {fitness}")
            print(f"Player Energy: {player_energy}, Enemy Energy: {enemy_energy}")
            print(f"Individual Gain: {individual_gain}")
            print(f"Best Weights: {best_solution}")
