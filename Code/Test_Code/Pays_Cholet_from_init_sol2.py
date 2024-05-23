import itertools
import multiprocessing
import random
import pickle
import os
import time
import numpy as np
from matplotlib import pyplot as plt
import math
import concurrent.futures


ACCELERATED_MUTATION_THRESHOLD = 1000
POPULATION_SIZE = 1000
WEIGHT_LIMIT = 5850
bad = 999999
ACCELERATED_MUTATION_NUMBER = 3
OLD_GENERATION = 200
N_PROCESSES = 4
MUTATION_RATE = 0.2

if os.getcwd() != "A:\CodePython\RechOp2024\Data\Probleme_Cholet_1_bis":
    os.chdir("../Data/Probleme_Cholet_1_bis/")
# Load necessary data
with open("init_sol_Cholet_pb1_bis.pickle", "rb") as f:
    init_solu = pickle.load(f)
    #pass

with open("dist_matrix_Cholet_pb1_bis.pickle", "rb") as f:
    dist_matrix = pickle.load(f)

with open("dur_matrix_Cholet_pb1_bis.pickle", "rb") as f:
    dur_matrix = pickle.load(f)

with open("temps_collecte_Cholet_pb1_bis.pickle", "rb") as f:
    collection_time = pickle.load(f)

with open("weight_Cholet_pb1_bis.pickle", "rb") as f:
    weight_list = pickle.load(f)

#init_solu = [0, 80, 81, 82, 83, 84, 210, 85, 86, 87, 88, 78, 89, 90, 212, 219, 213, 214, 216, 204, 211, 155, 177, 169, 172, 168, 167, 44, 45, 46, 47, 48, 49, 43, 136, 135, 134, 166, 34, 132, 131, 130, 35, 36, 37, 38, 39, 165, 29, 116, 67, 1, 2, 199, 164, 147, 4, 186, 10, 11, 12, 68, 191, 146, 187, 117, 69, 163, 158, 23, 189, 15, 127, 118, 159, 144, 143, 41, 13, 222, 14, 142, 227, 195, 62, 63, 54, 194, 196, 197, 6, 7, 8, 218, 198, 119, 193, 176, 64, 70, 20, 9, 145, 51, 52, 53, 65, 55, 56, 139, 57, 138, 129, 58, 59, 60, 61, 185, 141, 140, 50, 21, 22, 31, 32, 33, 71, 72, 66, 74, 205, 75, 97, 112, 152, 151, 226, 156, 209, 217, 215, 231, 114, 179, 180, 92, 157, 171, 228, 170, 110, 111, 183, 220, 161, 182, 91, 206, 230, 149, 173, 201, 174, 188, 73, 150, 190, 30, 181, 148, 3, 208, 202, 200, 137, 192, 120, 42, 175, 96, 5, 93, 121, 128, 94, 24, 25, 26, 27, 28, 126, 125, 124, 123, 95, 103, 221, 229, 104, 105, 106, 107, 16, 40, 122, 133, 17, 18, 19, 108, 178, 109, 77, 100, 101, 184, 115, 79, 162, 98, 99, 203, 154, 153, 102, 207, 160, 224, 113, 76, 225, 223, 232]


def initialize_population(init_sol, population_size):
    population = [init_sol.copy() for _ in range(population_size)]
    print(f"Génération 0: Solution initiale: {fitness(init_sol)} {init_sol}")
    for i in range(1, population_size):
        #if random.random() < MUTATION_RATE:
            population[i] = mutation(population[i].copy(), random.randint(1, 5))
    return population


def fitness(chemin) -> float:
    total_distance = 0
    total_weight = 0
    penalty = 0
    max_distance = 0
    index_max_distance = 0
    for i, j in itertools.islice(zip(chemin, chemin[1:]), len(chemin) - 1):
        if dist_matrix[i][j] > max_distance:
            max_distance = dist_matrix[i][j]
            index_max_distance = j
        total_distance += dist_matrix[i][j]
        total_weight += weight_list[i]
        total_weight = max(total_weight, 0)
        if total_weight > WEIGHT_LIMIT:
            penalty += bad
    return (total_distance + penalty, index_max_distance)

# def fitness_worker(sub_population):
#     return [fitness(individual) for individual in sub_population]
#
# # Fonction pour diviser la population et paralléliser le calcul
# def parallel_fitness(population):
#     pool = multiprocessing.Pool(N_PROCESSES)
#     # Diviser la population en sous-populations pour chaque processus
#     sub_populations = [population[i::N_PROCESSES] for i in range(N_PROCESSES)]
#     # Calculer la fitness de chaque sous-population en parallèle
#     fitness_values = pool.map(fitness_worker, sub_populations)
#     pool.close()
#     pool.join()
#     # Fusionner les résultats
#     return [item for sublist in fitness_values for item in sublist]
def fitness_worker(sub_population):
    # print(os.getcwd())
    if os.getcwd() != "A:\CodePython\RechOp2024\Data\Probleme_Cholet_1_bis":
        os.chdir("../Data/Probleme_Cholet_1_bis/")
    return [zip((fitness(individual) for individual in sub_population), sub_population)]

def parallel_fitness(population, n_processes):
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
        # Diviser la population en sous-populations pour chaque processus
        sub_populations = [population[i::n_processes] for i in range(n_processes)]
        # Calculer la fitness de chaque sous-population en parallèle
        fitness_values = executor.map(fitness_worker, sub_populations)
        # Fusionner les résultats
        return [item for sublist in fitness_values for item in sublist]

def selection(population):
    fitness_values = parallel_fitness(population, N_PROCESSES)
    #ranked_solutions = sorted([(fitness(s), s) for s in population], reverse=False)
    ranked_solutions = sorted(fitness_values, reverse=False, key=lambda x: x[0][0])
    #ranked_solutions = sorted([(fitness_val, ind) for fitness_val, ind in zip(fitness_values, population)],reverse=False)
    return ranked_solutions[:POPULATION_SIZE // 4]


def calculateVariance(population):
    mean = sum(sol[0][0] for sol in population) / len(population)
    return sum((sol[0][0] - mean) ** 2 for sol in population) / len(population)


def linear_base(variance, min_variance=100000, max_variance=2500000, min_base=0.0005, max_base=1.01):
    if variance <= min_variance:
        return min_base  # Grande base pour favoriser l'exploitation
    elif variance >= max_variance:
        return max_base  # Petite base pour favoriser l'exploration
    else:
        return max_base + (min_base - max_base) * ((variance - min_variance) / (max_variance - min_variance))


def tournament_selection(population, variance, tournament_size=10):
    n = len(population)
    # weights = [n - i for i in range(n)]
    base = linear_base(variance)  # exponential_base(variance)
    weights = [math.exp(-base * i) for i in range(n)]
    return min(random.choices(population, weights=weights, k=tournament_size))


def crossover(parent1: list[int], parent2: list[int], p1cp, p2cp) -> list[int]:

    #crossover_point = random.randint(len(parent1) // 4, 3 * len(parent1) // 4)
    child = [0]
    child_set = set(child)
    for j in range(1, len(parent1)//2):#p1cp):
        child.append(parent1[j])
        child_set.add(parent1[j])
    for element in parent2:
        if element not in child_set:
            child.append(element)
            child_set.add(element)
    return child


def mutation(individual, length=3):
    #if random.random() < MUTATION_RATE:
    start = random.randint(1, len(individual) - 2 - length)
    end = start + length
    segment = individual[start:end]
    del individual[start:end]
    new_position = random.randint(1, len(individual) - 2)
    individual = individual[:new_position] + segment + individual[new_position:]
    return individual

def genetic_algorithm(init_sol, population_size, best_scores, variances):
    population = initialize_population(init_sol, population_size)
    start_time = time.time()
    generation = 0
    stuck_generations = 0
    previous_score = 0
    while time.time() - start_time < 600:
        population = selection(population)
        if generation == 0:
            print(population[0][1])

        best_score = population[0][0][0]
        best_scores.append(best_score)

        if best_score == previous_score:
            stuck_generations += 1
        else:
            stuck_generations = 0
            previous_score = best_score

        generation += 1
        print(f"Generation {generation}: {best_score}", end=" ")

        population_variance = calculateVariance(population)
        print(f"Variance: {population_variance}")
        # variances.append(population_variance)
        #adjust_mutation_rate(population_variance)

        new_population = []

        new_population.extend(individual[1] for individual in population[:population_size // 50])

        while len(new_population) < population_size //2:
            parent1, parent2 = tournament_selection(population, population_variance), tournament_selection(population,
                                                                                                           population_variance)
            child = crossover(parent1[1], parent2[1], parent1[0][1], parent2[0][1])
            #if random.random() < MUTATION_RATE:
            #if population_variance == 0 and stuck_generations > 10:
            #    child = inversion_mutation(child, random.randint(1, 15))
            #else:
            if stuck_generations > 10:
                mutation_length = random.randint(2, 6)
            else:
                mutation_length = random.randint(1, 3)
            #else:
            #    child = mutation(child, random.randint(10, 15))
            child = mutation(child, mutation_length)
            new_population.append(child)

        while len(new_population) < population_size:
            parent1, parent2 = tournament_selection(population, population_variance), tournament_selection(population,
                                                                                                           population_variance)
            child = crossover(parent1[1], parent2[1], parent1[0][1], parent2[0][1])
            #if random.random() < MUTATION_RATE:
            if stuck_generations > 10:
                mutation_length = random.randint(6, 18)
            else:
                mutation_length = random.randint(3, 9)
            child = mutation(child, mutation_length)
            new_population.append(child)

        population = new_population

    return min(population, key=fitness)


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

if __name__ == "__main__":
    best_scores = []
    variances = []
    best_solution = genetic_algorithm(init_solu, POPULATION_SIZE, best_scores, variances)

    generation = [i for i in range(len(best_scores))]
    fitness = [s for s in best_scores]

    print(f"La meilleure solutions jamais obtenue est : {min(best_scores)}")

    print(best_solution)
    distance, temps = calculateDandT(best_solution)
    print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")
    print(f"Fitness: {distance + temps}")
    print(has_duplicates(best_solution))

    distance, temps = calculateDandT(init_solu)
    print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")
    print(f"fitness sol initiale : {distance + temps}")

    plt.scatter(generation, fitness)
    plt.show()