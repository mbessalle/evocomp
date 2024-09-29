from demo_controller import player_controller
import os
import matplotlib.pyplot as plt
import numpy as np
from evoman.environment import Environment

def create_box_plot(gains_ea1, gains_ea2, enemies):
    """Creates a box plot for individual gains."""
    # plt.figure()
    # plt.boxplot(gains)
    # plt.title(f'Box Plot of Individual Gains - Enemy {enemy}')
    # plt.ylabel('Gain')
    # plt.savefig(f'individual_gains_enemy_{enemy}.png')
    # plt.show()

    # Labels for each group
    labels = ['GA_E1', 'EDA_E1', 'GA_E2', 'EDA_E2', 'GA_E3', 'EDA_E3']

    data = [item for pair in zip(gains_ea1, gains_ea2) for item in pair]
    print(data)

    # Create the boxplot
    plt.figure(figsize=(8, 6))
    box = plt.boxplot(data,patch_artist=True)



    # Set the x-axis labels to the enemies
    plt.xticks([1,2,3,4,5,6], labels=labels)

    # # Customize box colors
    colors = ['steelblue', 'lightblue'] * 3  # Alternating colors
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Label the axes
    plt.xlabel("Enemies", fontsize=12)
    plt.ylabel("Individual Gain", fontsize=12)


    # Set y-axis limits from -100 to 100
    plt.ylim(-50, 110)

    # Title of the plot
    plt.title("Boxplot of Individual Gain for Different Enemies", fontsize=14)

    # Save the boxplot
    plt.savefig("plots/boxplot.png")

    # Show the plot
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





if __name__ == '__main__':

    os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = "weight_test"
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10
    enemies = [1,2,3]
    tries = 10




    # Dictionary to store individual gains for each enemy
    individual_gains = {enemy: [] for enemy in enemies}

    for attempt in range(1, tries + 1):
        for enemy in enemies:

            env = Environment(
                experiment_name=experiment_name,
                enemies=[enemy],
                playermode="ai",
                player_controller=player_controller(n_hidden_neurons),
                speed="fastest",
                enemymode="static",
                level=2,
                visuals=False,
            )

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

    best_sols = {}
    best_fitnesses = {}
    best_fitnesses_per_gen = {}
    mean_fitnesses_per_gen = {}
    eda_gains = []


    for enemy in enemies:
        best_sols[enemy] = []
        best_fitnesses[enemy] = []
        best_fitnesses_per_gen[enemy] = []
        mean_fitnesses_per_gen[enemy] = []
        for run in range(10):
            with open(f"eda_test/stats_enemy{enemy}_run_{run}", 'r') as file:
                lines = [line.strip().replace("np.float64(", "").replace("(", "").replace(")", "").split(",") for line
                         in file.readlines()]

            stats = list(zip(*lines))
            bests = [np.float64(stat) for stat in stats[0]]
            means = [np.float64(stat) for stat in stats[1]]

            best_fitnesses_per_gen[enemy].append(bests)
            mean_fitnesses_per_gen[enemy].append(means)
            best_fitnesses[enemy].append(stats[0][-1])

            best_sol = np.loadtxt(f"eda_test/best_eda_enemy_{enemy}_run_{run}.txt")

            best_sols[enemy].append(best_sol)

        headless = True
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        experiment_name = 'eda_test'
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        n_hidden_neurons = 10

        # initializes simulation in individual evolution mode, for single static enemy.
        env = Environment(experiment_name=experiment_name,
                          enemies=[enemy],
                          playermode="ai",
                          player_controller=player_controller(n_hidden_neurons),
                          # you  can insert your own controller here
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          visuals=False)

        eda_gains.append(test_best_solutions(env, best_sols[enemy]))

    create_box_plot(gains_to_plot,eda_gains,enemies)

    import numpy as np
    import matplotlib.pyplot as plt

    for enemy in enemies:
        runs = 10
        gens = 50

        fitness_own_ea = []

        for o in range(1, runs + 1):
            with open("results/best_fitness_level" + str(enemy) + "_run" + str(o)) as f:
                best_f = f.readlines()[:-1]

            for i in range(len(best_f)):
                best_f[i] = best_f[i].split(",")
                for j in range(len(best_f[i])):
                    best_f[i][j] = best_f[i][j].strip()
                    best_f[i][j] = float(best_f[i][j])

            fitness_own_ea.append(best_f)

        avg_best_f_own_ea = []
        for j in range(len(fitness_own_ea[0])):
            sum_fit = 0
            for e in range(len(fitness_own_ea)):
                sum_fit += fitness_own_ea[e][j][0]
            avg_best_f_own_ea.append(sum_fit / len(fitness_own_ea))
            if len(avg_best_f_own_ea) == gens:
                break

        mean_fitnesses_own_ea = []

        for o in range(1, runs + 1):
            with open("results/average_fitness_level" + str(enemy) + "_run" + str(o)) as f:
                mean_f = f.readlines()[:-1]

            for i in range(len(mean_f)):
                mean_f[i] = mean_f[i].split(",")
                for j in range(len(mean_f[i])):
                    mean_f[i][j] = mean_f[i][j].strip()
                    mean_f[i][j] = float(mean_f[i][j])

            mean_fitnesses_own_ea.append(mean_f)

        avg_mean_f_own_ea = []
        std_own_ea = []

        for j in range(len(mean_fitnesses_own_ea[0])):
            values_std = []
            sum_fit = 0

            for e in range(len(mean_fitnesses_own_ea)):
                sum_fit += mean_fitnesses_own_ea[e][j][0]
                values_std.append(mean_fitnesses_own_ea[e][j][0])

            std_own_ea.append(np.std(values_std))
            avg_mean_f_own_ea.append(sum_fit / len(mean_fitnesses_own_ea))

            if len(avg_mean_f_own_ea) == gens:
                break

        std_own_ea_plus = np.add(np.array(avg_mean_f_own_ea), np.array(std_own_ea))
        std_own_ea_minus = np.subtract(np.array(avg_mean_f_own_ea), np.array(std_own_ea))

        all_best_fitness_per_gen = np.array(best_fitnesses_per_gen[enemy])
        all_mean_fitness_per_gen = np.array(mean_fitnesses_per_gen[enemy])

        # Calculate mean and std over runs
        avg_best_fitness_eda = np.mean(all_best_fitness_per_gen, axis=0)
        std_best_fitness_eda = np.std(all_best_fitness_per_gen, axis=0)
        avg_mean_fitness_eda = np.mean(all_mean_fitness_per_gen, axis=0)
        std_mean_fitness_eda = np.std(all_mean_fitness_per_gen, axis=0)

        std_mean_eda_plus = np.add(np.array(avg_mean_fitness_eda), np.array(std_mean_fitness_eda))
        std_mean_eda_minus = np.subtract(np.array(avg_mean_fitness_eda), np.array(std_mean_fitness_eda))



        line1 = plt.plot(avg_best_f_own_ea, label="Avg best fitness GA")

        line2 = plt.plot(avg_mean_f_own_ea, label="Avg mean fitness GA", color="green")

        line3 = plt.plot(std_own_ea_plus, "--", linewidth=0.3)
        line4 = plt.plot(std_own_ea_minus, "--", linewidth=0.3)

        line5 = plt.plot(avg_best_fitness_eda, label="Avg best fitness EDA",color="red")

        line6 = plt.plot(avg_mean_fitness_eda, label="Avg mean fitness EDA", color="orange")

        line7 = plt.plot( std_mean_eda_plus , "--", linewidth=0.3)
        line8 = plt.plot(std_mean_eda_minus, "--", linewidth=0.3)





        np_gens = np.arange(0, gens, 1)
        np_gens_eda = np.arange(0, 30, 1)
        plt.fill_between(np_gens, std_own_ea_plus, std_own_ea_minus, alpha=0.3)

        plt.fill_between(np_gens_eda, std_mean_eda_plus, std_mean_eda_minus, alpha=0.1)

        plt.legend(prop={"family": "serif"})
        plt.xlabel("Generations", fontfamily="serif", fontsize=12)
        plt.ylabel("Fitness Value", fontfamily="serif", fontsize=12)
        plt.yticks(range(0, 101, 20))
        plt.title("Fitness vs. Generation, Enemy " + str(enemy), fontsize=16, pad=10, fontfamily="serif")
        plt.savefig("plots/plot_enemy" + str(enemy) + ".png")
        plt.show()


