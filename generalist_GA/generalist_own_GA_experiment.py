import numpy as np
import os
from evoman.environment import Environment
from demo_controller import player_controller
import random
import json

# Set number of hidden neurons
n_hidden_neurons = 10

# Initialize environment for enemies 2, 4, 6, 8 (generalist agent)
experiment_name = 'generalist_ga_experiment'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(experiment_name=experiment_name,
                  enemies=[2, 4, 6, 8],  # Subgroup of enemies
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# Evaluate the fitness of an individual
def evaluate(population):
    fitness = np.zeros(len(population))
    for i, individual in enumerate(population):
        fitness[i], _, _, _ = env.play(pcont=individual)
    return fitness

# Initialize population
def initialize_population(pop_size, n_vars, dom_l, dom_u):
    return np.random.uniform(dom_l, dom_u, (pop_size, n_vars))

# Tournament selection
def tournament_selection(population, fitness, tournament_size):
    selected = []
    for _ in range(2):
        candidates = np.random.choice(len(population), size=tournament_size)
        best_idx = np.argmax(fitness[candidates])
        selected.append(population[candidates[best_idx]])
    return selected[0], selected[1]

# Crossover
def crossover(parent1, parent2, crossover_rate):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(0, len(parent1))
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    return parent1.copy(), parent2.copy()

# Mutation
def mutate(individual, mutation_rate, dom_l, dom_u):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] += np.random.normal(0, 0.1)  # Gaussian mutation
            individual[i] = np.clip(individual[i], dom_l, dom_u)  # Keep within bounds
    return individual

# Run the genetic algorithm
def run_ga(pop_size, gens, mutation_rate, crossover_rate, tournament_size, dom_l, dom_u, run_id):
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5  # Number of weights
    population = initialize_population(pop_size, n_vars, dom_l, dom_u)
    best_fitness_per_gen = []
    mean_fitness_per_gen = []
    
    best_overall_fitness = -np.inf
    best_overall_individual = None
    
    for generation in range(gens):
        fitness = evaluate(population)
        
        best_fitness = np.max(fitness)
        mean_fitness = np.mean(fitness)
        best_individual = population[np.argmax(fitness)]
        
        if best_fitness > best_overall_fitness:
            best_overall_fitness = best_fitness
            best_overall_individual = best_individual

        best_fitness_per_gen.append(best_fitness)
        mean_fitness_per_gen.append(mean_fitness)
        
        print(f"Run {run_id} - Generation {generation}: Best fitness = {best_fitness}, Mean fitness = {mean_fitness}")
        
        # Create next generation
        next_population = []
        while len(next_population) < pop_size:
            parent1, parent2 = tournament_selection(population, fitness, tournament_size)
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            next_population.append(mutate(child1, mutation_rate, dom_l, dom_u))
            if len(next_population) < pop_size:
                next_population.append(mutate(child2, mutation_rate, dom_l, dom_u))
        
        population = np.array(next_population)
    
    # Save data for this run
    results_dir = os.path.join(experiment_name, f'run_{run_id}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save best weights for the run
    np.savetxt(os.path.join(results_dir, 'best_weights.txt'), best_overall_individual)

    # Save fitness history
    with open(os.path.join(results_dir, 'fitness_history.json'), 'w') as f:
        json.dump({'best_fitness_per_gen': best_fitness_per_gen, 'mean_fitness_per_gen': mean_fitness_per_gen}, f)
    
    return best_overall_fitness

# Run experiment 10 times
def run_experiment_10_times():
    pop_size = 100
    gens = 70
    mutation_rate = 0.3
    crossover_rate = 1.0
    tournament_size = 5
    dom_l = -1
    dom_u = 1
    
    best_fitness_all_runs = []
    
    for run_id in range(1, 11):
        print(f"Starting run {run_id}...")
        best_fitness = run_ga(pop_size, gens, mutation_rate, crossover_rate, tournament_size, dom_l, dom_u, run_id)
        best_fitness_all_runs.append(best_fitness)
    
    # Save best fitness for each run
    with open(os.path.join(experiment_name, 'best_fitness_all_runs.txt'), 'w') as f:
        f.write('\n'.join(map(str, best_fitness_all_runs)))

# Execute the experiment
run_experiment_10_times()
