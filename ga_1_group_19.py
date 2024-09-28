###############################################################################
# EvoMan Framework - Genetic Algorithm Implementation                         #
# Author: Group 19                                                           #
# Assignment: Task I - Specialist Agent Training using GA                     #
###############################################################################

# Imports framework
import sys
from evoman.environment import Environment
from demo_controller import player_controller

# Imports other libs
import numpy as np
import os
import matplotlib.pyplot as plt

# Set random seed for reproducibility (optional)
# np.random.seed(42)

# Runs simulation
def simulation(env, individual):
    """Runs the simulation and returns the fitness, player energy, and enemy energy."""
    fitness, player_energy, enemy_energy, time = env.play(pcont=individual)
    return fitness, player_energy, enemy_energy

# Evaluate the population
def evaluate(env, population):
    """Evaluates the fitness of the population."""
    fitnesses = []
    player_energies = []
    enemy_energies = []
    for individual in population:
        fitness, player_energy, enemy_energy = simulation(env, individual)
        fitnesses.append(fitness)
        player_energies.append(player_energy)
        enemy_energies.append(enemy_energy)
    return np.array(fitnesses), np.array(player_energies), np.array(enemy_energies)

# Tournament selection
def tournament_selection(pop, fitness_pop):
    """Performs tournament selection."""
    selected = np.random.choice(len(pop), 2)
    return pop[selected[0]] if fitness_pop[selected[0]] > fitness_pop[selected[1]] else pop[selected[1]]

# Crossover and mutation
def crossover_and_mutate(pop, fitness_pop, mutation_rate, dom_l, dom_u):
    """Performs crossover and mutation."""
    offspring = np.zeros((pop.shape[0], pop.shape[1]))
    for i in range(0, pop.shape[0], 2):
        parent1 = tournament_selection(pop, fitness_pop)
        parent2 = tournament_selection(pop, fitness_pop)
        cross_point = np.random.randint(1, pop.shape[1])
        offspring[i, :cross_point] = parent1[:cross_point]
        offspring[i, cross_point:] = parent2[cross_point:]
        offspring[i+1, :cross_point] = parent2[:cross_point]
        offspring[i+1, cross_point:] = parent1[cross_point:]

        # Mutation
        for child in [offspring[i], offspring[i+1]]:
            if np.random.rand() < mutation_rate:
                child += np.random.normal(0, 1, child.shape)
    return np.clip(offspring, dom_l, dom_u)

# Genetic Algorithm (GA) main function
def run_ga(env, pop_size, gens, mutation_rate, dom_l, dom_u, n_vars):
    """Runs the Genetic Algorithm."""
    best_solutions = []

    # Lists to store statistics per generation
    best_fitness_per_gen = []
    mean_fitness_per_gen = []

    # Initialize population randomly
    population = np.random.uniform(dom_l, dom_u, (pop_size, n_vars))

    # Evaluate the initial population
    fitness_pop, player_energies_pop, enemy_energies_pop = evaluate(env, population)

    for gen in range(gens):
        print(f"Generation {gen + 1}/{gens}")

        # Generate offspring through crossover and mutation
        offspring = crossover_and_mutate(population, fitness_pop, mutation_rate, dom_l, dom_u)

        # Evaluate offspring fitness
        fitness_offspring, player_energies_offspring, enemy_energies_offspring = evaluate(env, offspring)

        # Merge population and offspring
        population = np.vstack((population, offspring))
        fitness_pop = np.hstack((fitness_pop, fitness_offspring))

        # Select the best individuals to form the next generation
        survivors_idx = np.argsort(fitness_pop)[-pop_size:]
        population = population[survivors_idx]
        fitness_pop = fitness_pop[survivors_idx]

        # Store statistics
        best_fitness = np.max(fitness_pop)
        mean_fitness = np.mean(fitness_pop)

        best_fitness_per_gen.append(best_fitness)
        mean_fitness_per_gen.append(mean_fitness)

        # Store the best solution
        best_solutions.append(population[np.argmax(fitness_pop)])

    # Return the best solution found and statistics
    best_idx = np.argmax(fitness_pop)
    return population[best_idx], fitness_pop[best_idx], best_fitness_per_gen, mean_fitness_per_gen

# Function to run 10 independent experiments
def run_experiments(env, pop_size, gens, mutation_rate, dom_l, dom_u, n_vars, num_runs=10):
    """Runs 10 independent experiments."""
    all_best_solutions = []
    all_best_fitness = []
    all_best_fitness_per_gen = []
    all_mean_fitness_per_gen = []

    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        best_solution, best_fitness, best_fitness_per_gen, mean_fitness_per_gen = run_ga(
            env, pop_size, gens, mutation_rate, dom_l, dom_u, n_vars)
        all_best_solutions.append(best_solution)
        all_best_fitness.append(best_fitness)
        all_best_fitness_per_gen.append(best_fitness_per_gen)
        all_mean_fitness_per_gen.append(mean_fitness_per_gen)

    return all_best_solutions, all_best_fitness, all_best_fitness_per_gen, all_mean_fitness_per_gen

# Function to test best solutions 5 times and calculate individual gain with more detailed logging
def test_best_solutions(env, best_solutions, num_tests=5):
    """Tests each best solution 5 times and calculates the individual gain."""
    all_gains = []

    for solution_idx, solution in enumerate(best_solutions):
        gains = []
        print(f"\nTesting best solution {solution_idx + 1}...")

        for test_run in range(num_tests):
            fitness, player_energy, enemy_energy = simulation(env, solution)
            gain = player_energy - enemy_energy

            # Log detailed information for each test run
            print(f"Test run {test_run + 1}: Fitness = {fitness}, Player Energy = {player_energy}, Enemy Energy = {enemy_energy}, Gain = {gain}")
            
            if gain == 0:
                print(f"WARNING: Zero gain detected for solution {solution_idx + 1}, test run {test_run + 1}. Player likely failed early.")
            
            gains.append(gain)

        average_gain = np.mean(gains)
        all_gains.append(average_gain)
        print(f"Average gain for solution {solution_idx + 1}: {average_gain}")

    return all_gains


# Function to create box plots for gains
def create_box_plot(gains, enemy):
    """Creates a box plot for individual gains."""
    plt.figure()
    plt.boxplot(gains)
    plt.title(f'Box Plot of Individual Gains - Enemy {enemy}')
    plt.ylabel('Gain')
    plt.savefig(f'individual_gains_enemy_{enemy}.png')
    plt.show()

# Function to plot fitness over generations
def plot_fitness(all_best_fitness_per_gen, all_mean_fitness_per_gen, gens, enemy):
    """Plots average and std of best and mean fitness over generations."""
    generations = range(1, gens + 1)
    all_best_fitness_per_gen = np.array(all_best_fitness_per_gen)
    all_mean_fitness_per_gen = np.array(all_mean_fitness_per_gen)

    # Calculate mean and std over runs
    mean_best_fitness = np.mean(all_best_fitness_per_gen, axis=0)
    std_best_fitness = np.std(all_best_fitness_per_gen, axis=0)
    mean_mean_fitness = np.mean(all_mean_fitness_per_gen, axis=0)
    std_mean_fitness = np.std(all_mean_fitness_per_gen, axis=0)

    # Plot mean fitness
    plt.figure()
    plt.errorbar(generations, mean_mean_fitness, yerr=std_mean_fitness, label='Mean Fitness', fmt='-o')
    plt.errorbar(generations, mean_best_fitness, yerr=std_best_fitness, label='Best Fitness', fmt='-s')
    plt.title(f'Fitness over Generations - Enemy {enemy}')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'fitness_over_generations_enemy_{enemy}.png')
    plt.show()

# Main function
def main():
    # Choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'optimization_test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # Set up the environment for the enemy
    enemies = [3]  # You can loop over enemies as needed
    for enemy in enemies:
        print(f"\nRunning experiments for Enemy {enemy}...")
        env = Environment(experiment_name=experiment_name,
                          enemies=[enemy],
                          playermode="ai",
                          player_controller=player_controller(n_hidden_neurons),
                          enemymode="static",
                          level=2,  # Level set to 2 as per guidelines
                          contacthurt='player',  # Set contacthurt to 'player'
                          speed="fastest",
                          visuals=False)

        # Define number of weights for the network
        n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

        # Genetic Algorithm parameters
        pop_size = 100
        gens = 30
        mutation_rate = 0.2
        dom_u = 1
        dom_l = -1

        # Run 10 independent experiments
        best_solutions, best_fitness, all_best_fitness_per_gen, all_mean_fitness_per_gen = run_experiments(
            env, pop_size, gens, mutation_rate, dom_l, dom_u, n_vars)

        # Plot fitness over generations
        plot_fitness(all_best_fitness_per_gen, all_mean_fitness_per_gen, gens, enemy)

        # Test best solutions 5 times and calculate individual gains
        gains = test_best_solutions(env, best_solutions)

        # Create box plot for gains
        create_box_plot(gains, enemy)

if __name__ == '__main__':
    main()
