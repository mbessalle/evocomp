###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import heapq
import os
import time
import matplotlib.pyplot as plt
import random

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

def select_parents(fit_pop, k):
    top_k_indices = [i for i, _ in heapq.nlargest(k, enumerate(fit_pop), key=lambda x: x[1])]
    best_parent_index = np.argmax(fit_pop)

    return top_k_indices, best_parent_index

def add_min_variance(cov_mat,min_var):
    for j in range(cov_mat.shape[0]):
        if cov_mat[j, j] < min_var:
            cov_mat[j, j] = min_var
    return cov_mat

def get_fitness_stats(fit_pop):
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    return best, mean, std

def generate_offspring(parents, min_var, npop, k, mean_converged):
    parents_mean = np.mean(parents, axis=0)
    parents_cov = np.cov(parents, rowvar=False)

    if mean_converged:
        new_cov = parents_cov.copy()
        # add min variance to cov matrix
        for j in range(parents_cov.shape[0]):
            if parents_cov[j, j] < min_var:
                new_cov[j, j] = min_var


        offspring = np.random.multivariate_normal(parents_mean,new_cov, npop - 1)
    else:
        offspring = np.random.multivariate_normal(parents_mean, parents_cov, npop - k)
    return offspring


# def migration(islands, env):
#
#     new_islands = [[0]] * len(islands)
#     island_fitnesses = [[0]] * len(islands)
#     for i, pop in enumerate(islands):
#         migrants = pop[:len(pop)//2].copy()
#         if i == len(islands)-1:
#             old_pop = islands[0][len(pop) // 2:].copy()
#             new_pop = np.vstack((old_pop, migrants))
#             new_islands[0] = new_pop
#             island_fitnesses[0] = evaluate(env,new_pop)
#         else:
#             old_pop = islands[i-1][len(pop)//2:].copy()
#             new_pop = np.vstack((old_pop, migrants))
#             new_islands[i + 1] = new_pop
#             island_fitnesses[i + 1] = evaluate(env, new_pop)
#
#
#
#
#     return island_fitnesses, new_islands


def migration(islands, env,n_pop):
    num_islands = len(islands)
    new_islands = [[0]] * num_islands
    new_island_fitnesses = [[0]] * num_islands
    for i in range(num_islands):
        migrants = []
        for j in range(num_islands):
            migrant_indices = random.sample(range(100),n_pop//num_islands)
            migrants.append(islands[j][migrant_indices])

        new_pop = migrants[0]
        for k in range(1,len(migrants)):
            new_pop = np.vstack((new_pop, migrants[k]))
        new_islands[i] = new_pop
        new_island_fitnesses[i] = evaluate(env,new_pop)
    return new_island_fitnesses, new_islands


def run_eda(enemy):
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    experiment_name = 'eda_test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=enemy,
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False,
                    multiplemode="yes")


    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5


    ini = time.time()

    # start writing your own code from here
    npop = 100
    min_var = 1e-5
    gens = 30
    k = 10
    n_islands = 3
    migration_frequency = 1

    lower_bounds = [-1,-2,-5]
    upper_bounds = [1,2,5]

    mean_converged = True

    fitness_stats = []
    islands = []
    island_fitnesses = []
    island_bests_indices = []
    island_bests_fitnesses = []
    island_means = []

    # create island populations
    for i in range(n_islands):
        # initialize population
        pop = np.random.uniform(low=lower_bounds[i], high=upper_bounds[i], size=(npop, n_vars))
        # pop = np.random.multivariate_normal(np.zeros(n_vars), np.eye(n_vars)*2, size=npop*10)

        islands.append(pop)
        # save initial fitness
        island_fitness = evaluate(env,pop)
        island_fitnesses.append(island_fitness)
        best, mean, std = get_fitness_stats(island_fitness)
        island_bests_indices.append(best)
        island_bests_fitnesses.append(island_fitness[best])
        island_means.append(mean)

        print('\n GENERATION ' + str(0) + ' ' + "ISLAND "+ str(i) + " " + str(round(island_bests_fitnesses[i], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
            round(std, 6)))

    fitness_stats.append((np.max(island_bests_fitnesses), mean, std))

    # initial pop at gen 0
    #pop = np.random.multivariate_normal(np.zeros(n_vars), np.eye(n_vars), size=npop*10)




    for i in range(gens-1):
        new_island_fitnesses = []
        new_islands = []
        for j, (island_fitness, pop) in enumerate(zip(island_fitnesses,islands)):

            # select parents for next gen
            parent_indices, best_parent_idx = select_parents(island_fitness,k)

            # if island_bests_fitnesses[j] - island_means[j] <= 1.0:
            #      mean_converged = True
            # else:
            #      mean_converged = False


            parents = pop[parent_indices]
            best_parent = pop[best_parent_idx]

            # generate offspring
            offsprings = generate_offspring(parents, min_var, npop, k, mean_converged)

            # elitism/survivor selection
            if mean_converged:
                pop = np.vstack((offsprings, best_parent))
            else:
                pop = np.vstack((offsprings, parents))

            new_islands.append(pop)
            new_island_fitness = evaluate(env, pop)
            new_island_fitnesses.append(new_island_fitness)
            best, mean, std = get_fitness_stats(new_island_fitness)
            island_bests_indices[j] = best
            island_bests_fitnesses[j] = new_island_fitness[best]
            island_means[j] = mean

            print('\n GENERATION ' + str(i+1) + ' ' + "ISLAND " + str(j) + " " + str(
                round(new_island_fitness[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))


        islands = new_islands.copy()
        island_fitnesses = new_island_fitnesses.copy()


        if i%migration_frequency == 0 and i !=0 and i != len(islands)-1:
            island_fitnesses, islands = migration(islands,env, npop)
            for m, island_fitness in enumerate(island_fitnesses):
                best, mean, std = get_fitness_stats(island_fitness)
                island_bests_indices[m] = best
                island_bests_fitnesses[m] = island_fitness[best]
                island_means[m] = mean

                print('\n GENERATION ' + str(i + 1) + ' POST-MIGRATION ' + "ISLAND " + str(m) + " " + str(
                    round(island_fitness[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
                 round(std, 6)))
        fitness_stats.append((np.max(island_bests_fitnesses), mean, std))


    fim = time.time()
    run_time = fim - ini
    best_idx = np.argmax(island_bests_fitnesses)
    best_individual_idx = island_bests_indices[best_idx]

    return fitness_stats, islands[best_idx][best_individual_idx], run_time

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
    plt.savefig(f'ea_2_eda_fitness_over_generations_enemy_{enemy}.png')
    plt.show()


def test_best_solutions(env, best_solutions, num_tests=5):
    """Tests each best solution 5 times and calculates the individual gain."""
    all_gains = []

    for solution_idx, solution in enumerate(best_solutions):
        gains = []
        print(f"\nTesting best solution {solution_idx + 1}...")

        for test_run in range(num_tests):
            fitness, player_energy, enemy_energy, _ = env.play(solution)
            gain = player_energy - enemy_energy

            # Log detailed information for each test run
            print(
                f"Test run {test_run + 1}: Fitness = {fitness}, Player Energy = {player_energy}, Enemy Energy = {enemy_energy}, Gain = {gain}")

            if gain == 0:
                print(
                    f"WARNING: Zero gain detected for solution {solution_idx + 1}, test run {test_run + 1}. Player likely failed early.")

            gains.append(gain)

        average_gain = np.mean(gains)
        all_gains.append(average_gain)
        print(f"Average gain for solution {solution_idx + 1}: {average_gain}")

    return all_gains

def create_box_plot(gains, enemy):
    """Creates a box plot for individual gains."""
    plt.figure()
    plt.boxplot(gains)
    plt.title(f'Box Plot of Individual Gains - Enemy {enemy}')
    plt.ylabel('Gain')
    plt.savefig(f'ea_2_eda_individual_gains_enemy_{enemy}.png')
    plt.show()


if __name__ == '__main__':

    # add other set of enemies
    enemies = [[1,2,5,8]]

    best_sols = {}
    best_fitnesses = {}
    best_fitnesses_per_gen = {}
    mean_fitnesses_per_gen = {}
    std_fitnesses_per_gen = {}
    run_times = {}

    for i, enemy in enumerate(enemies):
        best_sols[i] = []
        best_fitnesses[i] = []
        best_fitnesses_per_gen[i] = []
        mean_fitnesses_per_gen[i] = []
        std_fitnesses_per_gen[i] = []
        run_times[i] = []

        for run in range(10):
            fitness_stats, best_sol, run_time = run_eda(enemy)

            with open(f"eda_generalist_test/stats_enemies{enemy}_run_{run}", 'w') as file:
                for item in fitness_stats:
                    file.write(f"{item}\n")

            # saves file with the best solution
            np.savetxt(f'eda_generalist_test/best_eda_enemies{enemy}_run_{run}.txt', best_sol)

            best_sols[i].append(best_sol)
            best_fitnesses[i].append(fitness_stats[-1][0])
            stats = list(zip(*fitness_stats))
            best_fitnesses_per_gen[i].append(stats[0])
            mean_fitnesses_per_gen[i].append(stats[1])
            #std_fitnesses_per_gen[enemy].append(stats[2])
            run_times[i].append(run_time)

        #TODO
        plot_fitness(best_fitnesses_per_gen[i], mean_fitnesses_per_gen[i],30,enemy)

        # choose this for not using visuals and thus making experiments faster
        headless = True
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        experiment_name = 'eda_test'
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        n_hidden_neurons = 10

        # initializes simulation in individual evolution mode, for single static enemy.
        env = Environment(experiment_name=experiment_name,
                          enemies=enemies,
                          playermode="ai",
                          player_controller=player_controller(n_hidden_neurons),
                          # you  can insert your own controller here
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          visuals=False,
                          multiplemode="yes")

        gains = test_best_solutions(env, best_sols[i])

        create_box_plot(gains, i)
        break








