import random
import pickle
import os
import time
import numpy as np

os.chdir("../Data/Probleme_Cholet_1_bis/")

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

POPULATION_SIZE = 100
WEIGHT_LIMIT = 5850

def initialize_population(init_sol, population_size):
    population = [init_sol.copy() for _ in range(population_size)]
    for individual in population:
        subset = individual[1:-1]
        random.shuffle(subset)
        individual[1:-1] = subset

    return population


def fitness(individual):
    distance = 0
    time = 0
    weight = 0
    penalty = 1000000000

    for i in range(len(individual)):
        weight += weight_list[individual[i]]
        if weight > WEIGHT_LIMIT:
            return 1 / penalty
        if i != len(individual) - 1:
            distance += dist_matrix[individual[i]][individual[i + 1]]
            time += dur_matrix[individual[i]][individual[i + 1]]
        time += collection_time[individual[i]]
    return 1 / (distance + time)

def selection(population):
    return sorted(population, key=fitness, reverse=True)[:POPULATION_SIZE // 10]

#def crossover(parent1, parent2, crossover_rate=0.7):
#    if random.random() < crossover_rate:
#        cp = random.randint(1, len(parent1) - 2)
#        child1 = parent1[:cp] + parent2[cp:]
#        child_set = set(child1)
#        for i in range(1, len(child1) - 1):
#            if child1.count(child1[i]) > 1:
#                for j in range(1, len(parent2) - 1):
#                    if parent2[j] not in child_set:
#                        child_set.remove(child1[i])
#                        child_set.add(parent2[j])
#                        child1[i] = parent2[j]
#                        break
#        return child1
#    else:
#        return parent1

def crossover(parent1, parent2, crossover_rate=0.7):
    if random.random() <  crossover_rate:
        start = random.randint(1, len(parent1) - 2)
        end = random.randint(start, len(parent1) - 2)
        new_genes = [0]
        new_genes =  new_genes + parent1[start:end]
        for i in range(1, len(parent2) - 1):
            p = parent2[i]
            if p not in new_genes:
                new_genes.append(p)
        new_genes.append(232)
        return new_genes
    else : 
        return random.choice([parent1, parent2])

def mutation(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        index1, index2 = random.sample(range(1, len(individual) - 1), 2)
        individual[index1], individual[index2] = individual[index2], individual[index1]
    return individual

def genetic_algorithm(init_sol, population_size):
    population = initialize_population(init_sol, population_size)
    start_time = time.time()
    generation = 0
    while time.time() - start_time < 600:
        population = selection(population)
        new_population = []

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)
            child = mutation(child)
            new_population.append(child)

        population = new_population
        best_score= fitness(max(population, key=fitness))
        generation += 1
        print(f"Generation {generation}: {best_score}")

    return max(population, key=fitness)



def calculateDandT(l):
    distance = 0
    time = 0
    for i in range(len(l)):
        if i != len(l) - 1:
            distance += dist_matrix[l[i]][l[i + 1]]
            time += dur_matrix[l[i]][l[i + 1]]
        time += collection_time[l[i]]
    return distance , time / 3600

def has_duplicates(lst):
    return len(lst) != len(set(lst))


best_solution = genetic_algorithm(init_solu, POPULATION_SIZE)
print(best_solution)

print(has_duplicates(best_solution))
distance, time = calculateDandT(best_solution)
print(f"Distance: {distance} km, Temps: {time} h")

#distance, temps= calculateDandT(init_solu)
#print(f"Distance: {distance} , Temps: {temps} ")
