import numpy as np
import random
import os
import time
import pickle
from matplotlib import pyplot as plt


# Constants
ACCELERATED_MUTATION_THRESHOLD = 1000
POPULATION_SIZE = 500
WEIGHT_LIMIT = 5850
bad = 999999
ACCELERATED_MUTATION_NUMBER = 3
OLD_GENERATION = 200

os.chdir("../Data/Probleme_Cholet_1_bis/")

# Load necessary data
with open("init_sol_Cholet_pb1_bis.pickle", "rb") as f:
    init_solu = np.array(pickle.load(f))

with open("dist_matrix_Cholet_pb1_bis.pickle", "rb") as f:
    dist_matrix = np.array(pickle.load(f))

with open("dur_matrix_Cholet_pb1_bis.pickle", "rb") as f:
    dur_matrix = np.array(pickle.load(f))

with open("temps_collecte_Cholet_pb1_bis.pickle", "rb") as f:
    collection_time = np.array(pickle.load(f))

with open("weight_Cholet_pb1_bis.pickle", "rb") as f:
    weight_list = np.array(pickle.load(f))

# Functions
def initialize_population(init_sol, population_size):
    population = np.tile(init_sol, (population_size, 1))
    for individual in population:
        np.random.shuffle(individual[1:-1])
    return population

def custom_cumsum(x):
    custom_cumsum.current_sum = max(0, custom_cumsum.current_sum + x)
    return custom_cumsum.current_sum


def fitness(chemin):
    custom_cumsum.current_sum = 0
    total_distance = np.sum(dist_matrix[chemin[:-1], chemin[1:]])
    total_time = np.sum(dur_matrix[chemin[:-1], chemin[1:]]) + np.sum(collection_time[chemin])
    vfunc = np.vectorize(custom_cumsum)
    cumulative_weights = vfunc(weight_list[chemin])
    penalty = np.sum(np.where(cumulative_weights > WEIGHT_LIMIT, bad, 0))
    total_time += collection_time[chemin[-1]]
    return total_distance + total_time + penalty

def selection(population):
    fitness_values = np.array([fitness(s) for s in population])
    sorted_indices = np.argsort(fitness_values)
    return population[sorted_indices[:POPULATION_SIZE // 10]]


def tournament_selection(population, tournament_size=3):
    n = len(population)
    weights = np.arange(n, 0, -1)
    selected_indices = np.random.choice(n, size=tournament_size, p=weights/np.sum(weights))
    selected_fitness_values = np.array([fitness(elem) for elem in population[selected_indices]])
    return population[selected_indices[np.argmin(selected_fitness_values)]]


#def crossover(parent1, parent2):
#    parent1, parent2  = np.array(parent1), np.array(parent2)
#    child = np.zeros_like(parent1)
#    child[:len(parent1)//2] = parent1[:len(parent1)//2]
#    not_in_child = ~np.isin(parent2, child)
#    child[len(parent1)//2:] = parent2[not_in_child]
#    return child#.tolist()
def crossover(parent1, parent2):
    child = [0]
    child = np.array(child)
    for j in range(1, len(parent1) // 2):
        child.append(parent1[j])
    for element in parent2:
        if element not in child:
            child.append(element)
    return child

def mutation(individual, length=3):
    start = np.random.randint(1, len(individual) - 2 - length)
    end = start + length
    segment = individual[start:end]
    individual = np.delete(individual, segment)
    new_position = np.random.randint(1, len(individual) - 2)
    individual = np.insert(individual, new_position, segment)
    return individual

#def genetic_algorithm(init_sol, population_size, best_scores):
#    population = initialize_population(init_sol, population_size)
#    start_time = time.time()
#    generation = 0
#    while time.time() - start_time < 600:
#        population = selection(population)
#
#        best_score= fitness(population[0])
#        best_scores.append(best_score)
#
#        generation += 1
#        print(f"Generation {generation}: {best_score}")
#
#        new_population = []
#        new_population = np.array(new_population)
#
#        while len(new_population) < population_size:
#            parent1, parent2 = tournament_selection(population), tournament_selection(population)
#            child = crossover(parent1, parent2)
#            child = mutation(child, random.randint(1, 5))
#            new_population.append(child.tolist())
#
#    return new_population
def genetic_algorithm(init_sol, population_size):
    population = initialize_population(init_sol, population_size)
    best_scores = []
    start_time = time.time()

    while time.time() - start_time < 600:
        population = selection(population)
        best_score = fitness(population[0])
        best_scores.append(best_score)
        print(f"Generation {len(best_scores)}: {best_score}")

        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = tournament_selection(population), tournament_selection(population)
            child = crossover(parent1, parent2)
            child = mutation(child, np.random.randint(1, 6))
            new_population.append(child)

        population = np.array(new_population)

    return min(population, key=fitness), best_scores


best, best_scores  = genetic_algorithm(init_solu, POPULATION_SIZE)