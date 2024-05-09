import random
import pickle
import os
import time
import numpy as np
from matplotlib import pyplot as plt
import math
import itertools

ACCELERATED_MUTATION_THRESHOLD = 1000
POPULATION_SIZE = 500
WEIGHT_LIMIT = 5850
bad = 999999
ACCELERATED_MUTATION_NUMBER = 3
OLD_GENERATION = 50

os.chdir("../Data/Probleme_Cholet_1_bis/")

# Load necessary data
with open("init_sol_Cholet_pb1_bis.pickle", "rb") as f:
    init_solu = pickle.load(f)

np.savetxt('init_solu.txt', np.array(init_solu), fmt='%d')

with open("dist_matrix_Cholet_pb1_bis.pickle", "rb") as f:
    dist_matrix = pickle.load(f)
np.savetxt('dist_matrix.txt', np.array(dist_matrix), fmt='%f')

with open("dur_matrix_Cholet_pb1_bis.pickle", "rb") as f:
    dur_matrix = pickle.load(f)
np.savetxt('dur_matrix.txt', np.array(dur_matrix), fmt='%f')

with open("temps_collecte_Cholet_pb1_bis.pickle", "rb") as f:
    collection_time = pickle.load(f)
np.savetxt('collection_time.txt', np.array(collection_time), fmt='%f')

with open("weight_Cholet_pb1_bis.pickle", "rb") as f:
    weight_list = pickle.load(f)


def initialize_population(init_sol, population_size) -> list[tuple[int, list[int]]]:
    population = [(0, init_sol.copy()) for _ in range(population_size)]
    for age, individual in population:
        subset = individual[1:-1]
        random.shuffle(subset)
        # subset = mutation(subset)
        individual[1:-1] = subset

    return population


def fitness(chemin, age) -> float:
    total_distance = 0
    total_weight = 0
    total_time = 0
    penalty = 0
    for i, j in itertools.islice(zip(chemin, chemin[1:]), len(chemin) - 1):
        total_distance += dist_matrix[i][j]
        total_time += dur_matrix[i][j]
        total_time += collection_time[i]
        total_weight += weight_list[i]
        total_weight = max(total_weight, 0)
        if total_weight > WEIGHT_LIMIT:
            penalty += bad
    total_time += collection_time[chemin[-1]]
    if age > OLD_GENERATION:
        return (total_distance + total_time + penalty) * (1 + 0.01 * age)
    return total_distance + total_time + penalty


def selection(population: list[tuple[int, list[int]]]) -> list[tuple[float, int, list[int]]]:
    ranked_solutions = sorted([(fitness(s[1], s[0]), s[0] + 1, s[1]) for s in population], reverse=False, key=lambda x: x[0])
    return ranked_solutions[:POPULATION_SIZE // 10]


def calculateVariance(population) -> float:
    mean = sum(sol[0] for sol in population) / len(population)
    return sum((sol[0] - mean) ** 2 for sol in population) / len(population)


def linear_base(variance, min_variance=100000, max_variance=1000000, min_base=0.0005, max_base=1.01) -> float:
    if variance <= min_variance:
        return max_base  # Grande base pour favoriser l'exploitation
    elif variance >= max_variance:
        return min_base  # Petite base pour favoriser l'exploration
    else:
        return max_base + (min_base - max_base) * ((variance - min_variance) / (max_variance - min_variance))


def tournament_selection(population, variance, tournament_size=10) -> tuple[float, int, list[int]]:
    n = len(population)
    # weights = [n - i for i in range(n)]
    base = linear_base(variance)  # exponential_base(variance)
    weights = [math.exp(-base * i) for i in range(n)]
    # weights[1:] = [w*2 for w in weights[1:]]
    # if variance > 230000:
    #    weights = [math.exp(-0.4 * i) for i in range(n)]
    # else:
    #    weights = [pow(1.05, -i) for i in range(n)]
    return min(random.choices(population, weights=weights, k=tournament_size))


def crossover(parent1: list[int], parent2: list[int]) -> list[int]:
    child = [0]
    child_set = set(child)  # Create a set to track elements in child
    for j in range(1, len(parent1) // 2):
        child.append(parent1[j])
        child_set.add(parent1[j])  # Add new element to set
    for element in parent2:
        if element not in child_set:  # Check set for membership
            child.append(element)
            child_set.add(element)  # Add new element to set
    return child


def mutation(individual, length=3):
    start = random.randint(1, len(individual) - 2 - length)
    end = start + length
    segment = individual[start:end]
    del individual[start:end]
    new_position = random.randint(1, len(individual) - 2)
    individual = individual[:new_position] + segment + individual[new_position:]
    return individual


def genetic_algorithm(init_sol, population_size, best_scores, variances, carried_over_proportion=0.2):
    population: list[tuple[int, list[int]]] = initialize_population(init_sol, population_size)
    start_time = time.time()
    generation = 0
    while time.time() - start_time < 600:
        population: list[tuple[float, int, list[int]]] = selection(population)
        # population = [(fitness, age, individu), ...]

        best_score = population[0]
        best_scores.append(best_score)

        generation += 1
        print(f"Generation {generation}: {best_score[0]}, age {population[0][1]}", end=" ")

        population_variance = calculateVariance(population)
        print(f"Variance: {population_variance}")
        # variances.append(population_variance)

        new_population = []

        new_population.extend((individual[1], individual[2]) for individual in
                              population[:int(round(len(population) * carried_over_proportion, 0))])

        while len(new_population) < population_size // 2:
            parent1, parent2 = tournament_selection(population, population_variance), tournament_selection(population,
                                                                                                           population_variance)
            child = crossover(parent1[2], parent2[2])
            child = mutation(child, random.randint(1, 5))
            new_population.append((0, child))

        while len(new_population) < population_size:
            parent1, parent2 = tournament_selection(population, population_variance), tournament_selection(population,
                                                                                                           population_variance)
            child = crossover(parent1[2], parent2[2])
            child = mutation(child, random.randint(10, 15))
            new_population.append((0, child))

        population = new_population

    return min(population, key=lambda x: fitness(x[1], x[0]))


def calculateDandT(l):
    distance = 0
    time = 0
    for i in range(len(l)):  # [0, 232]
        if i != len(l) - 1:
            distance += dist_matrix[l[i]][l[i + 1]]
            time += dur_matrix[l[i]][l[i + 1]]
        time += collection_time[l[i]]
    return distance, time


def has_duplicates(lst):
    return len(lst) != len(set(lst))


best_scores = []
variances = []
best_solution = genetic_algorithm(init_solu, POPULATION_SIZE, best_scores, variances)

generation = [i for i in range(len(best_scores))]
fitness = [s[0] for s in best_scores]

print(f"La meilleure solutions jamais obtenue est : {min(best_scores)[0]}")

print(best_solution[1])
distance, temps = calculateDandT(best_solution[1])
print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")
print(f"Fitness: {distance + temps}")
print(has_duplicates(best_solution[1]))

distance, temps = calculateDandT(init_solu)
print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")
print(f"fitness sol initiale : {distance + temps}")

print(f"Age moyen des meilleures solutions : {sum([s[1] for s in best_scores]) / len(best_scores)}")

plt.scatter(generation, fitness)
plt.show()
