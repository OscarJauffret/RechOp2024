import random
import pickle
import os
import time
import numpy as np
from matplotlib import pyplot as plt
import math

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
        #subset = mutation(subset)
        individual[1:-1] = subset

    return population

def fitness(chemin):
    total_distance = 0
    total_weight = 0
    total_time = 0
    penalty = 0
    for i in range(len(chemin) - 1): #[0, 231]
        total_distance += dist_matrix[chemin[i]][chemin[i + 1]]

        total_time += dur_matrix[chemin[i]][chemin[i + 1]]
        total_time += collection_time[chemin[i]]

        total_weight += weight_list[chemin[i]]
        total_weight = max(total_weight, 0)
        if total_weight > WEIGHT_LIMIT:
            penalty += bad
            
    total_time += collection_time[chemin[-1]]
    return total_distance + total_time + penalty

def selection(population):
    ranked_solutions = sorted([(fitness(s), s) for s in population], reverse=False)
    return ranked_solutions[:POPULATION_SIZE//10]

def calculateVariance(population):
    mean = sum(sol[0] for sol in population) / len(population)
    return sum((sol[0] - mean) ** 2 for sol in population) / len(population)

def calculate_base(variance):
    """ Calcule une base de décroissance dynamique en fonction de la variance. """
    if variance > 230000:
        return 0.3 + (variance / 1000000)  # Exemple de formule adaptative
    else:
        return 1.05  # Base plus stable pour des variances faibles

def tournament_selection(population, variance, tournament_size=3):
    n = len(population)
    #weights = [n - i for i in range(n)]
    base = calculate_base(variance)
    weights = [math.exp(-base * i) for i in range(n)]
    #if variance > 230000:
    #    weights = [math.exp(-0.4 * i) for i in range(n)]
    #else:
    #    weights = [pow(1.05, -i) for i in range(n)]
    return min(random.choices(population, weights=weights, k=tournament_size))


def crossover(parent1, parent2):
    child = [0]
    for j in range(1, len(parent1) // 2):
        child.append(parent1[j])
    for element in parent2:
        if element not in child:
            child.append(element)
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
    while time.time() - start_time < 600:
        population = selection(population)

        best_score= population[0][0]
        best_scores.append(best_score)

        generation += 1
        print(f"Generation {generation}: {best_score}", end=" ")

        population_variance = calculateVariance(population)
        print(f"Variance: {population_variance}")
        #variances.append(population_variance)

        new_population = []

        while len(new_population) < population_size:
            parent1, parent2 = tournament_selection(population, population_variance), tournament_selection(population, population_variance)
            #child1, child2 = crossover(parent1[0], parent1[1], parent2[0], parent2[1])
            child = crossover(parent1[1],parent2[1])
            #child1, child2 = inversion_mutation(child1, random.randint(1, 5)), inversion_mutation(child2, random.randint(1, 5))
            child = mutation(child, random.randint(1, 5))
            new_population.append(child)
            #new_population.append(child2)

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

generation = [i for i in range(len(best_scores))]
fitness = [s for s in best_scores]

print(best_solution)
distance, temps = calculateDandT(best_solution)
print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")
print(f"Fitness: {distance + temps}")
print(has_duplicates(best_solution))


distance, temps= calculateDandT(init_solu)
print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")
print(f"fitness sol initiale : {distance + temps}")

plt.scatter(generation, fitness)
plt.show()

#generation = [i for i in range(len(best_scores))]
#variance = [s for s in variances]
#
#plt.scatter(generation, variance)
#plt.show()


#def calculate_mass(parent):
#    mass = []
#    for i in range(len(parent)):
#        left_neighbor = parent[i - 1] if i > 0 else -1
#        right_neighbor = parent[i + 1] if i < len(parent) - 1 else -1
#        if left_neighbor == -1 :
#            
#        mass.append(dist_matrix[parent[i]][left_neighbor] + dist_matrix[parent[i]][right_neighbor])
#    return mass
#
#def calculate_velocity(parent):
#    return sum(dist_matrix[parent[i]][parent[i + 1]] for i in range(len(parent) - 1)) / len(parent)
#
#def collision(m1, m2, v1, v2):
#    v1_prime = ((m1 - m2) / (m1 + m2)) * v1 + ((2 * m2) / (m1 + m2)) * v2
#    v2_prime = ((2 * m1) / (m1 + m2)) * v1 - ((m1 - m2) / (m1 + m2)) * v2
#    return v1_prime, v2_prime
#
#def crossover(fitness1,parent1, fitness2, parent2):
#    offspring1 = parent1.copy()
#    offspring2 = parent2.copy()
#
#    #total_velocity1 = calculate_velocity(parent1)
#    #total_velocity2 = calculate_velocity(parent2)
#    for i in range(len(parent1)):
#        mass1 = 0
#        mass2 = 0
#        left_neighbor1 = parent1[i - 1] if i > 0 else -1
#        right_neighbor1 = parent1[i + 1] if i < len(parent1) - 1 else -1
#        if left_neighbor1 != -1:
#            mass1 += dist_matrix[parent1[i]][left_neighbor1]
#        if right_neighbor1 != -1:
#            mass1 += dist_matrix[parent1[i]][right_neighbor1]
#
#        left_neighbor2 = parent2[i - 1] if i > 0 else -1
#        right_neighbor2 = parent2[i + 1] if i < len(parent2) - 1 else -1
#        if left_neighbor2 != -1:
#            mass2 += dist_matrix[parent2[i]][left_neighbor2]
#        if right_neighbor2 != -1:
#            mass2 += dist_matrix[parent2[i]][right_neighbor2]
#
#        v1_prime, v2_prime = collision(mass1, mass2, fitness1, fitness2)#total_velocity1, total_velocity2)
#        if v1_prime <= 0:
#            offspring1[i] = parent2[i]
#        if v2_prime <= 0:
#            offspring2[i] = parent1[i]
#
#    return offspring1, offspring2