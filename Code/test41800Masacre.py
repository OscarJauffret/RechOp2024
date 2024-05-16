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
STUCK_GEN_MAX = 850
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
    print(f"Génération 0: Solution initiale: {fitness(init_sol)} {init_sol}")
    population = [init_sol.copy()]
    for _ in range(1, population_size):
        new_individual = mutation(init_sol.copy(), random.randint(1, 5))
        population.append(new_individual)
    return population


def fitness(chemin) -> tuple[float, int]:
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
    return total_distance + penalty, index_max_distance


def selection(population, pool):
    fitness_values = pool.map(fitness, population)
    ranked_solutions = sorted(zip(fitness_values, population), reverse=False)
    return ranked_solutions[:POPULATION_SIZE // 6]


def calculateVariance(population):
    mean = sum(sol[0][0] for sol in population) / len(population)
    return sum((sol[0][0] - mean) ** 2 for sol in population) / len(population)


# NB : le 41.8km a été obtenu en utilisant max_variance = 1_000_000 et min_variance = 100_000
def linear_base(variance, min_variance=100000, max_variance=200000, min_base=0.0005, max_base=1.01):
    if variance <= min_variance:
        return min_base     # Petite base pour favoriser l'exploration
    elif variance >= max_variance:
        return max_base     # Grande base pour favoriser l'exploitation, donc on donne plus de poids aux meilleures solutions
    else:
        return max_base + (min_base - max_base) * ((variance - min_variance) / (max_variance - min_variance))


def tournament_selection(population, variance, tournament_size=10):
    n = len(population)
    base = linear_base(variance)
    weights = [math.exp(-base * i) for i in range(n)]
    return min(random.choices(population, weights=weights, k=tournament_size))


def crossover(parent1: list[int], parent2: list[int], p1cp: int, p2cp:  int) -> list[int]:
    # Définir les segments en coupant les parents sur leurs pires gènes
    if p1cp < p2cp:
        segment1 = parent1[1:p1cp]
        segment2 = parent2[p1cp:p2cp]
        segment3 = parent1[p2cp:]
    else:
        segment1 = parent2[1:p2cp]
        segment2 = parent1[p2cp:p1cp]
        segment3 = parent2[p1cp:]

    # Initialiser l'enfant avec le premier élément commun
    child = [0]
    child_set = set(child)

    # Ajouter les segments à l'enfant tout en évitant les doublons
    for segment in [segment1, segment2, segment3]:
        for gene in segment:
            if gene not in child_set:
                child.append(gene)
                child_set.add(gene)

    # Ajouter les éléments restants des parents pour compléter l'enfant
    for parent in [parent1, parent2]:
        for gene in parent:
            if gene not in child_set:
                child.append(gene)
                child_set.add(gene)
    return child


def mutation(individual, length=3):
    start = random.randint(1, len(individual) - 2 - length)
    end = start + length
    segment = individual[start:end]
    del individual[start:end]
    new_position = random.randint(1, len(individual) - 2)
    individual = individual[:new_position] + segment + individual[new_position:]
    return individual

#def swap_mutation(individual):
#    i, j = random.sample(range(1, len(individual) - 1), 2)
#    individual[i], individual[j] = individual[j], individual[i]
#    return individual


def genetic_algorithm(init_sol, population_size, best_scores):
    global MUTATION_RATE
    original_mutation_rate = MUTATION_RATE  # Enregistrer la valeur originale de MUTATION_RATE
    population = initialize_population(init_sol, population_size)
    start_time = time.time()
    generation = 0
    stuck_generations = 0
    previous_score = 0
    with multiprocessing.Pool() as pool:
        while time.time() - start_time < 6:
            population = selection(population, pool)

            best_score = population[0][0][0]
            best_scores.append((best_score,population[0][1]))

            if best_score == previous_score:
                stuck_generations += 1
            else:
                stuck_generations = 0
                #print("Je me reset")
                previous_score = best_score

            generation += 1
            #print(f"Generation {generation}: {best_score}", end=" ")

            population_variance = calculateVariance(population)
            #print(f"Variance: {population_variance}")
            variances.append(population_variance)

            new_population = []
            if stuck_generations >= STUCK_GEN_MAX:
            # Réinitialisation partielle : introduire des nouveaux individus aléatoires
                #print("Massacre en douceur")
                num_new_individuals = int(POPULATION_SIZE // 1.2)
                new_population.extend(initialize_population(init_sol, num_new_individuals))
                new_population.extend(individual[1] for individual in population[num_new_individuals:])
                MUTATION_RATE = 1.0  # Augmenter le taux de mutation à 100% pour cette génération
                #print("MASACREEEEEEEEEEEE")
                #new_population = initialize_population(init_sol, POPULATION_SIZE)
            elif stuck_generations >= 50: 
                #print("Recadrement")
                population=population[50:]  #Delete the 50 best solutions
                new_population.extend(individual[1] for individual in population) 
            else:
                new_population.extend(individual[1] for individual in population[:population_size // 50]) 

            while len(new_population) < population_size // 2:
                parent1, parent2 = tournament_selection(population, population_variance), tournament_selection(population, population_variance)
                child = crossover(parent1[1], parent2[1], parent1[0][1], parent2[0][1])
                #mutation_length = random.randint(2, 6) if stuck_generations > 10 else random.randint(1, 3)
                mutation_length = random.randint(20, 50) if stuck_generations > STUCK_GEN_MAX else random.randint(2,6)
                if random.random() < MUTATION_RATE:
                    #if rand := random.randint(1, 3) == 1:
                    child = mutation(child, mutation_length)
                    #child = three_opt_mutation(child)
                    #elif rand == 2:
                #child = mutation(child, mutation_length)
                new_population.append(child)

            while len(new_population) < population_size:
                parent1, parent2 = tournament_selection(population, population_variance), tournament_selection(
                    population, population_variance)
                child = crossover(parent1[1], parent2[1], parent1[0][1], parent2[0][1])
                #mutation_length = random.randint(6, 18) if stuck_generations > 10 else random.randint(3, 9)
                mutation_length = random.randint(60, 90) if stuck_generations > STUCK_GEN_MAX else random.randint(6, 18)
                if random.random() < MUTATION_RATE:
                    #if rand := random.randint(1, 3) == 1:
                    child = mutation(child, mutation_length)
                    #child = three_opt_mutation(child)
                    #elif rand == 2:
                    #child = swap_mutation(child)
                new_population.append(child)

            population = new_population
            MUTATION_RATE = original_mutation_rate  # Réinitialiser MUTATION_RATE à sa valeur originale

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

def ispermutation(l):
    return sorted(l) == list(range(233)) and l[0] == 0 and l[-1] == 232


if __name__ == "__main__":
    best_scores = []
    variances = []
    best_solution = genetic_algorithm(init_solu, POPULATION_SIZE, best_scores)

    generation = [i for i in range(len(best_scores))]
    fitness = [s[0] for s in best_scores]

    #print(f"La meilleure solutions jamais obtenue est : {min(best_scores)[0]}, avec ce chemin : {min(best_scores)[1]}")

    print(min(best_scores)[0])
    #print(best_solution)
    #distance, temps = calculateDandT(best_solution)
    #print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")
    #print(f"Fitness: {distance + temps}")
    #print(has_duplicates(best_solution))
    #print(f"Est-ce une bonne solution ? {ispermutation(best_solution)}")
#
    #distance, temps = calculateDandT(init_solu)
    #print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")
    #print(f"fitness sol initiale : {distance + temps}")
#
    #plt.scatter(generation, fitness)
    #plt.show()
