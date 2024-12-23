import random
import pickle
import os
import time
import numpy as np
from matplotlib import pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tournament_size", type=int, help="The size of the tournament")
args = parser.parse_args()

if args.tournament_size:
    TOURNAMENT_SIZE = args.tournament_size

ACCELERATED_MUTATION_THRESHOLD = 1000
POPULATION_SIZE = 500
WEIGHT_LIMIT = 5850
bad = 999999
ACCELERATED_MUTATION_NUMBER = 3
OLD_GENERATION = 200

os.chdir("../Data/Probleme_Cholet_1_bis/")

import numpy as np

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


def initialize_population(init_sol, population_size):
    population = [init_sol.copy() for _ in range(population_size)]
    for individual in population:
        subset = individual[1:-1]
        random.shuffle(subset)
        # subset = mutation(subset)
        individual[1:-1] = subset

    return population


def fitness(chemin):
    total_distance = 0
    total_weight = 0
    total_time = 0
    penalty = 0
    for i in range(len(chemin) - 1):  # [0, 231]
        total_distance += dist_matrix[chemin[i]][chemin[i + 1]]
        # total_distance += dist_matrix[chemin[i]][chemin[i]]

        # total_time += dur_matrix[chemin[i]][chemin[i]]
        total_time += dur_matrix[chemin[i]][chemin[i + 1]]
        total_time += collection_time[chemin[i]]

        total_weight += weight_list[chemin[i]]

        if total_weight > WEIGHT_LIMIT:
            penalty += bad

    # total_time += dur_matrix[chemin[-1]][chemin[-1]]
    # total_distance += dist_matrix[chemin[-1]][chemin[-1]]
    total_time += collection_time[chemin[-1]]
    return total_distance + total_time + penalty


def selection(population):
    ranked_solutions = sorted([(fitness(s), s) for s in population], reverse=False)
    return ranked_solutions[:POPULATION_SIZE // 10]


def tournament_selection(population, tournament_size=3):
    n = len(population)
    weights = [n - i for i in range(n)]
    return random.choices(population, weights=weights, k=tournament_size)[0]


def crossover(parent1, parent2):
    child = [0]
    for j in range(1, len(parent1) // 2):
        child.append(parent1[j])
    for element in parent2:
        if element not in child:
            child.append(element)
    return child


# def mutation(individual):
#    number_of_mutations = random.randint(5, 15)
#    for _ in range(number_of_mutations):
#        index = random.randint(1, len(individual) - 2)
#        individual[index], individual[index + 1] = individual[index + 1], individual[index]
#    return individual

def inversion_mutation(individual, length=3):
    start = random.randint(1, len(individual) - 2 - length)
    # end = random.randint(start, len(individual) - 2)
    end = start + length
    segment = individual[start:end]
    del individual[start:end]
    new_position = random.randint(1, len(individual) - 2)
    individual = individual[:new_position] + segment + individual[new_position:]
    return individual


def genetic_algorithm(init_sol, population_size, best_scores):
    population = initialize_population(init_sol, population_size)
    start_time = time.time()
    generation = 0
    while time.time() - start_time < 600:
        population = selection(population)

        best_score = population[0][0]
        best_scores.append(best_score)

        # nb_of_mutations = calculate_number_of_mutations(best_scores)

        generation += 1
        #print(f"Generation {generation}: {best_score}")

        new_population = []

        while len(new_population) < population_size:
            if TOURNAMENT_SIZE >= 11:
                parent1, parent2 = tournament_selection(population, random.randint(1, 10)), tournament_selection(population,
                                                                                                         random.randint(1, 10))
            else:
                parent1, parent2 = tournament_selection(population, TOURNAMENT_SIZE), tournament_selection(population, TOURNAMENT_SIZE)
            child = crossover(parent1[1], parent2[1])
            child = inversion_mutation(child, random.randint(1, 5))
            new_population.append(child)

        population = new_population

    return fitness(population[0])


def calculate_number_of_mutations(best_scores):
    if len(best_scores) > OLD_GENERATION and (
            abs(best_scores[-1] - best_scores[-1 - OLD_GENERATION]) < ACCELERATED_MUTATION_THRESHOLD):
        print("Accelerated mutation", end=" ")
        return ACCELERATED_MUTATION_NUMBER
    else:
        print("Normal mutation", end=" ")
        return 1


def calculateDandT(l):
    distance = 0
    time = 0
    for i in range(len(l)):  # [0, 232]
        if i != len(l) - 1:
            distance += dist_matrix[l[i]][l[i + 1]]
            time += dur_matrix[l[i]][l[i + 1]]
        # distance += dist_matrix[l[i]][l[i]]
        # time += dur_matrix[l[i]][l[i]]
        time += collection_time[l[i]]
    return distance, time


def has_duplicates(lst):
    return len(lst) != len(set(lst))


best_scores = []
best_solution = genetic_algorithm(init_solu, POPULATION_SIZE, best_scores)

generation = [i for i in range(len(best_scores))]
fitness = [s for s in best_scores]

print(best_solution)

#print(best_solution)
#distance, temps = calculateDandT(best_solution)
#print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")
#print(f"Fitness: {distance + temps}")
#print(has_duplicates(best_solution))
#
#distance, temps = calculateDandT(init_solu)
#print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")
#print(f"fitness sol initiale : {distance + temps}")

#plt.scatter(generation, fitness)
#plt.show()