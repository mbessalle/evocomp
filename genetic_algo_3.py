from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import random
import os


def run_simulation(environment, individual, minimize=False):

    fitness, player_life, enemy_life, time = environment.play(pcont=individual)
    if not minimize:
        return fitness
    else:
        normalized_fitness = (fitness - (-100)) / (100 - (-100)) * 100
        return abs(normalized_fitness - 100)


def evaluate_population(environment, population):

    return np.array([run_simulation(environment, ind) for ind in population])


def select_parents(num_best, population, fitness_scores):

    return population[np.argpartition(np.array(fitness_scores), -num_best)[-num_best:]]


def apply_mutation(parents_group, mutation_rate, leave_local_optimum):
    """Gaussian noise mutation"""
    mutated_offspring = []
    for parent in parents_group:
        # duplicate parent
        mutated_parent = parent.copy()

        # iterate over the genes and randomly add gaussian noise
        for i in range(len(parent)):
            if random.uniform(0, 1) < mutation_rate:
                if leave_local_optimum:
                    mutation_value = mutated_parent[i] + np.random.normal(0, 10)
                else:
                    mutation_value = mutated_parent[i] + np.random.normal(0, 1)
                mutated_parent[i] = mutation_value

        mutated_offspring.append(mutated_parent)

    return np.array(mutated_offspring)


def perform_crossover(mutated_group, num_best, total_population_size):
    """ Random Crossover"""
    offspring = []
    for _ in range(total_population_size - num_best):
        # select two random parents
        parent_a, parent_b = random.sample(list(mutated_group), 2)

        # generate random crossover index
        crossover_index = random.randint(1, len(parent_a))

        # concatenate parts of both parents to create offspring
        offspring.append(np.concatenate((parent_a[:crossover_index], parent_b[crossover_index:])))

    return offspring


def select_survivors(mutated_offspring, offspring, current_population, fitness_scores, num_best, total_population_size, environment, replacements):
    """
    Modified Elitism selection strategy.
    """
    worst_individuals = current_population[np.argpartition(np.array(fitness_scores), (total_population_size - num_best))[(total_population_size - num_best):]]
    next_generation = list(mutated_offspring)

    for child in offspring:
        # evaluate child against a random individual
        random_individual = random.choice(worst_individuals)
        fitness_child = run_simulation(environment, child)
        fitness_random = run_simulation(environment, random_individual)

        # replace random individual if child is better
        if fitness_child > fitness_random:
            next_generation.append(child)
            replacements += 1
        else:
            next_generation.append(random_individual)

    new_population = np.array(next_generation)
    fitness_new_population = evaluate_population(environment, new_population)
    
    return new_population, fitness_new_population, replacements


def main():
    # parameters
    enemy_level = 3
    num_runs = 10
    num_best_individuals = 10
    population_size = 40
    num_generations = 50

    print("\nStarting the simulation...")
    print("Enemy Level:", str(enemy_level))
    print("Number of Runs:", str(num_runs))
    print("Population Size:", str(population_size))
    print("Generations:", str(num_generations))

    for run in range(1, num_runs + 1):
        print("\nRun Number: " + str(run))

        headless_mode = True
        if headless_mode:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        experiment_directory = "genetic_algorithm_3"

        hidden_neurons = 10

        # initialize environment
        environment = Environment(
            experiment_name=experiment_directory,
            enemies=[enemy_level],
            playermode="ai",
            player_controller=player_controller(hidden_neurons),
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False,
        )

        # number of weights for the neural network
        num_variables = (environment.get_num_sensors() + 1) * hidden_neurons + (hidden_neurons + 1) * 5

        # GA parameters
        upper_bound = 1
        lower_bound = -1
        mutation_rate = 0.2
        replacements_count = 0
        stuck_in_local_optimum = False
        population = np.random.uniform(lower_bound, upper_bound, (population_size, num_variables))
        population_fitness = evaluate_population(environment, population)
        best_fitness = [np.amax(np.array(population_fitness))]
        average_fitness = [np.mean(np.array(population_fitness))]

        # Evolution loop
        for generation in range(num_generations + 1):
            # track best and mean fitness
            best_fitness.append(np.amax(np.array(population_fitness)))
            average_fitness.append(np.mean(np.array(population_fitness)))

            # print generation details
            if generation % 5 == 0:
                print("########################")
                print("GENERATION: ", generation)
                print("BEST FITNESS: ", np.amax(np.array(population_fitness)))
                print("AVERAGE FITNESS: ", np.mean(np.array(population_fitness)))

            # Selection
            top_parents = select_parents(num_best_individuals, population, population_fitness)

            # Mutation
            mutated_offspring = apply_mutation(top_parents, mutation_rate, stuck_in_local_optimum)

            # Crossover
            new_offspring = perform_crossover(mutated_offspring, num_best_individuals, population_size)

            # Survivor selection
            population, population_fitness, replacements_count = select_survivors(
                mutated_offspring, new_offspring, population, population_fitness, num_best_individuals, population_size, environment, replacements_count
            )

            # Handle local optimum
            if generation % 10 == 0 and generation != 0 and best_fitness.count(best_fitness[-1]) > 10:
                if replacements_count < 5:
                    stuck_in_local_optimum = True
                    print("-----TRYING TO EXIT LOCAL OPTIMUM-----")
                else:
                    stuck_in_local_optimum = False
                replacements_count = 0

        # save results
        np.savetxt("results/population_level" + str(enemy_level) + "_run" + str(run), population, delimiter=",")
        np.savetxt("results/fitness_level" + str(enemy_level) + "_run" + str(run), population_fitness, delimiter=",")
        np.savetxt("results/best_fitness_level" + str(enemy_level) + "_run" + str(run), best_fitness, delimiter=",")
        np.savetxt("results/average_fitness_level" + str(enemy_level) + "_run" + str(run), average_fitness, delimiter=",")


if __name__ == "__main__":
    main()
