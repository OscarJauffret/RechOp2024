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
OLD_GENERATION = 200

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


def initialize_population(init_sol, population_size):
    population = [init_sol.copy() for _ in range(population_size)]
    for i, individual in enumerate(population[1:]):
        if i < population_size // 2:
            population[i] = mutation(init_sol.copy(), random.randint(1, 5))
        else:
            for _ in range(2):
                population[i] = mutation(population[i], random.randint(10, 15))

    return population

def fitness(chemin):
    total_distance = 0
    total_weight = 0
    total_time = 0
    penalty = 0
    for i, j in itertools.islice(zip(chemin, chemin[1:]), len(chemin) - 1):
        total_distance += dist_matrix[i][j]
        #total_time += dur_matrix[i][j]
        #total_time += collection_time[i]
        total_weight += weight_list[i]
        total_weight = max(total_weight, 0)
        if total_weight > WEIGHT_LIMIT:
            penalty += bad
            
    #total_time += collection_time[chemin[-1]]
    return total_distance + total_time + penalty

def selection(population):
    ranked_solutions = sorted([(fitness(s), s) for s in population], reverse=False)
    return ranked_solutions[:POPULATION_SIZE//10]

def calculateVariance(population):
    mean = sum(sol[0] for sol in population) / len(population)
    return sum((sol[0] - mean) ** 2 for sol in population) / len(population)


def linear_base(variance, min_variance=100000, max_variance=1000000, min_base=0.0005, max_base=1.01):
    if variance <= min_variance:
        return max_base     # Grande base pour favoriser l'exploitation
    elif variance >= max_variance:
        return min_base     # Petite base pour favoriser l'exploration
    else:
        return max_base + (min_base - max_base) * ((variance - min_variance) / (max_variance - min_variance))   


def tournament_selection(population, variance, tournament_size=10):
    n = len(population)
    base = linear_base(variance)#exponential_base(variance)
    weights = [math.exp(-base * i) for i in range(n)]

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
    if random.random() < 0.5:
        segment = segment[::-1]
    del individual[start:end]
    new_position = random.randint(1, len(individual) - 2)
    individual = individual[:new_position] + segment + individual[new_position:]
    return individual


def genetic_algorithm(init_sol, population_size, best_scores, variances):
    population = initialize_population(init_sol, population_size)
    start_time = time.time()
    generation = 0
    while time.time() - start_time < 600:
        population = selection(population)

        best_score = population[0][0]
        best_score = population[0][0]
        best_scores.append(best_score)

        generation += 1

        population_variance = calculateVariance(population)

        #if generation%500 == 0 or generation == 1:
        # print(f"Generation {generation}: {best_score}", end=" ")
        # print(f"Variance: {population_variance}")
            

        new_population = []

        #new_population.extend(individual[1] for individual in population[:population_size//4])
        new_population.append(population[0][1])

        while len(new_population) < population_size//2:
            parent1, parent2 = tournament_selection(population, population_variance), tournament_selection(population, population_variance)
            child = crossover(parent1[1],parent2[1])
            child = mutation(child, random.randint(1, 5))
            new_population.append(child)

        while len(new_population) < population_size:
            parent1, parent2 = tournament_selection(population, population_variance), tournament_selection(population, population_variance)
            child = crossover(parent1[1], parent2[1])
            child = mutation(child, random.randint(7, 10))
            new_population.append(child)

        population = new_population

    return min(population, key=fitness)



def calculateDandT(l):
    distance = 0
    time = 0
    for i in range(len(l)):#[0, 232]
        if i != len(l) - 1:
            distance += dist_matrix[l[i]][l[i + 1]]
            time += dur_matrix[l[i]][l[i + 1]]
        time += collection_time[l[i]]
    return distance , time 

def has_duplicates(lst):
    return len(lst) != len(set(lst))


best_scores = []
variances = []
best_solution = genetic_algorithm(init_solu, POPULATION_SIZE, best_scores, variances)

#generation = [i for i in range(len(best_scores))]
#fitness = [s for s in best_scores]

#print(f"La meilleure solutions jamais obtenue est : {min(best_scores)}")
#print(f"en {len(generation)} générations")

#print(best_solution)
distance, temps = calculateDandT(best_solution)
#print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")
print(f"{distance + temps}, {distance}, {best_solution}")
#print(has_duplicates(best_solution))


#distance, temps= calculateDandT(init_solu)
#print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")
#print(f"fitness sol initiale : {distance + temps}")

#plt.scatter(generation, fitness)
#plt.show()
