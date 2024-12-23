import itertools
import multiprocessing
import random
import pickle
import os
import time
import numpy as np
from matplotlib import pyplot as plt
import math

ACCELERATED_MUTATION_THRESHOLD = 1000
POPULATION_SIZE = 1000
WEIGHT_LIMIT = 5850
bad = 999999
ACCELERATED_MUTATION_NUMBER = 3
OLD_GENERATION = 200
MUTATION_RATE = 0.9


try:
    os.chdir("../Data/Probleme_Cholet_1_bis/")
except FileNotFoundError:
    pass

# Load necessary data
with open("init_sol_Cholet_pb1_bis.pickle", "rb") as f:
    init_solu = pickle.load(f)

with open("dist_matrix_Cholet_pb1_bis.pickle", "rb") as f:
    dist_matrix = pickle.load(f)

with open("dur_matrix_Cholet_pb1_bis.pickle", "rb") as f:
    dur_matrix = pickle.load(f)

with open("temps_collecte_Cholet_pb1_bis.pickle", "rb") as f:
    collection_time = pickle.load(f)

with open("weight_Cholet_pb1_bis.pickle", "rb") as f:
    weight_list = pickle.load(f)


def initialize_population(init_sol, population_size):
    population = [init_sol.copy() for _ in range(population_size)]
    print(f"Génération 0: Solution initiale: {fitness(init_sol)} {init_sol}")
    for i in range(1, population_size):
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


def selection(population, pool):
    fitness_values = pool.map(fitness, population)
    ranked_solutions = sorted(zip(fitness_values, population), reverse=False)
    return ranked_solutions[:POPULATION_SIZE // 4]


def calculateVariance(population):
    mean = sum(sol[0][0] for sol in population) / len(population)
    return sum((sol[0][0] - mean) ** 2 for sol in population) / len(population)


def linear_base(variance, min_variance=100000, max_variance=2500000, min_base=0.0005, max_base=1.01):
    if variance <= min_variance:
        return min_base
    elif variance >= max_variance:
        return max_base
    else:
        return max_base + (min_base - max_base) * ((variance - min_variance) / (max_variance - min_variance))


def tournament_selection(population, variance, tournament_size=10):
    n = len(population)
    base = linear_base(variance)
    weights = [math.exp(-base * i) for i in range(n)]
    return min(random.choices(population, weights=weights, k=tournament_size))


def crossover(parent1: list[int], parent2: list[int], p1cp, p2cp) -> list[int]:
    child = [0]
    child_set = set(child)
    for j in range(1, len(parent1) // 2):
        child.append(parent1[j])
        child_set.add(parent1[j])
    for element in parent2:
        if element not in child_set:
            child.append(element)
            child_set.add(element)
    return child


def mutation(individual, length=3):
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
    with multiprocessing.Pool() as pool:
        while time.time() - start_time < 600:
            population = selection(population, pool)
            if generation == 0:
                print(population[0][1])

            best_score = population[0][0][0]
            best_scores.append((best_score,population[0][1]))

            if best_score == previous_score:
                stuck_generations += 1
            else:
                stuck_generations = 0
                print("Je me reset")
                previous_score = best_score

            generation += 1
            print(f"Generation {generation}: {best_score}", end=" ")

            population_variance = calculateVariance(population)
            print(f"Variance: {population_variance}")
            variances.append(population_variance)

            new_population = []
            if stuck_generations >= 400:
                print("MASACREEEEEEEEEEEE")
                population=population[50:]
                new_population.extend(individual[1] for individual in population) 
            elif stuck_generations >= 50: 
                new_population.extend(individual[1] for individual in population[:population_size // 50]) 


            #if generation < 2000:
            #    new_population.extend(individual[1] for individual in population[:population_size // 50])

            while len(new_population) < population_size // 2:
                parent1, parent2 = tournament_selection(population, population_variance), tournament_selection(population, population_variance)
                child = crossover(parent1[1], parent2[1], parent1[0][1], parent2[0][1])
                #mutation_length = random.randint(20, 50) if stuck_generations > 400 else random.randint(2,6)
                mutation_length = random.randint(2,6) if stuck_generations > 10 else random.randint(1,3)
                child = mutation(child, mutation_length)
                new_population.append(child)

            while len(new_population) < population_size:
                parent1, parent2 = tournament_selection(population, population_variance), tournament_selection(
                    population, population_variance)
                child = crossover(parent1[1], parent2[1], parent1[0][1], parent2[0][1])
                #mutation_length = random.randint(60, 90) if stuck_generations > 400 else random.randint(6, 18)
                mutation_length = random.randint(6, 18) if stuck_generations > 10 else random.randint(3, 9)
                child = mutation(child, mutation_length)
                new_population.append(child)

            population = new_population

    return min(population, key=fitness)


def calculateDandT(l):
    distance = 0
    time = 0
    for i in range(len(l)):
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
    fitness = [s[0] for s in best_scores]

    print(f"La meilleure solutions jamais obtenue est : {min(best_scores)[0]}, avec ce chemin : {min(best_scores)[1]}")

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
