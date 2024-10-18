import numpy as np
import os
from evoman.environment import Environment
from demo_controller import player_controller
import random

# Set the number of hidden neurons
n_hidden_neurons = 10

# Initialize the environment for generalist training
experiment_name = 'generalist_ga_hyperopt'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Initialize environment for multiple enemies (generalist agent)
env = Environment(experiment_name=experiment_name,
                  enemies=[2, 4, 6, 8],  # Generalist agent fights multiple enemies
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

# Initialization of the population
def initialize_population(pop_size, n_vars, dom_l, dom_u):
    return np.random.uniform(dom_l, dom_u, (pop_size, n_vars))

# Tournament selection
def tournament_selection(population, fitness, tournament_size):
    selected = []
    for _ in range(2):  # Two individuals for crossover
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

# Main GA loop
def run_ga(pop_size, gens, mutation_rate, crossover_rate, tournament_size, dom_l, dom_u):
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5  # Total weights
    population = initialize_population(pop_size, n_vars, dom_l, dom_u)
    best_fitness_per_gen = []
    
    for generation in range(gens):
        fitness = evaluate(population)
        
        # Track the best individual
        best_fitness = np.max(fitness)
        best_individual = population[np.argmax(fitness)]
        best_fitness_per_gen.append(best_fitness)
        print(f"Generation {generation}: Best fitness = {best_fitness}")
        
        # Create the next generation
        next_population = []
        while len(next_population) < pop_size:
            parent1, parent2 = tournament_selection(population, fitness, tournament_size)
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            next_population.append(mutate(child1, mutation_rate, dom_l, dom_u))
            if len(next_population) < pop_size:
                next_population.append(mutate(child2, mutation_rate, dom_l, dom_u))
        
        population = np.array(next_population)
    
    return best_individual, best_fitness_per_gen

# Hyperparameter search space
def random_search(num_trials=10):
    search_space = {
        "pop_size": [50, 100, 200],        # Population sizes
        "gens": [30, 50, 70],              # Number of generations
        "mutation_rate": [0.1, 0.2, 0.3],  # Mutation rates
        "crossover_rate": [0.6, 0.8, 1.0], # Crossover rates
        "tournament_size": [3, 5, 7],      # Tournament size
        "dom_l": [-1, -0.5],               # Lower bound for weights
        "dom_u": [0.5, 1]                  # Upper bound for weights
    }
    
    best_fitness = -np.inf
    best_hyperparams = None
    
    for trial in range(num_trials):
        # Randomly sample hyperparameters
        params = {k: random.choice(v) for k, v in search_space.items()}
        
        print(f"Running trial {trial+1} with hyperparameters: {params}")
        
        # Run GA with the sampled hyperparameters
        best_individual, fitness_history = run_ga(params['pop_size'], params['gens'], 
                                                  params['mutation_rate'], params['crossover_rate'], 
                                                  params['tournament_size'], params['dom_l'], 
                                                  params['dom_u'])
        
        # Get the best fitness from this trial
        trial_best_fitness = max(fitness_history)
        
        # Check if this is the best solution
        if trial_best_fitness > best_fitness:
            best_fitness = trial_best_fitness
            best_hyperparams = params
        
        print(f"Trial {trial+1} best fitness: {trial_best_fitness}")
    
    print(f"Best hyperparameters: {best_hyperparams} with fitness: {best_fitness}")
    return best_hyperparams, best_fitness

# Run the hyperparameter optimization
best_hyperparams, best_fitness = random_search(num_trials=10)
print(f"Best hyperparameters found: {best_hyperparams}")
