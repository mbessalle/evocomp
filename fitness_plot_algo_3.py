import numpy as np
import matplotlib.pyplot as plt


enemy = 3
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

line1 = plt.plot(avg_best_f_own_ea, label="Avg best fitness Algo 3")

line3 = plt.plot(avg_mean_f_own_ea, label="Avg mean fitness Algo 3", color="green")

line5 = plt.plot(std_own_ea_plus, "--", linewidth=0.3)
line6 = plt.plot(std_own_ea_minus, "--", linewidth=0.3)

np_gens = np.arange(0, gens, 1)
plt.fill_between(np_gens, std_own_ea_plus, std_own_ea_minus, alpha=0.3)

plt.legend(prop={"family":"serif"})
plt.xlabel("Generations", fontfamily="serif", fontsize=12)
plt.ylabel("Fitness Value", fontfamily="serif", fontsize=12)
plt.yticks(range(0, 101, 20))
plt.title("Fitness vs. Generation, Enemy " + str(enemy), fontsize=16, pad=10, fontfamily="serif")
plt.savefig("plots/plot_enemy" + str(enemy) + ".png")
plt.show()