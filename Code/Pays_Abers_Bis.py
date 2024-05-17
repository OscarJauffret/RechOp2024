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
    os.chdir("../Data/Probleme_Abers_2_bis/")
except FileNotFoundError:
    pass
# Load necessary data
with open("init_sol_Abers_pb2_bis.pickle", "rb") as f:
    init_solu = pickle.load(f)
    init_solu = init_solu.transpose()
    init_solu = init_solu.values.tolist()
    init_solu = [item for sublist in init_solu for item in sublist]

with open("dist_matrix_Abers_pb2_bis.pickle", "rb") as f:
    dist_matrix = pickle.load(f)

with open("weight_Abers_pb2_bis.pickle", "rb") as f:
    weight_list = pickle.load(f)

with open("dur_matrix_Abers_pb2_bis.pickle", "rb") as f:
    dur_matrix = pickle.load(f)

with open("temps_collecte_Abers_pb2_bis.pickle", "rb") as f:
    collection_time = pickle.load(f)

with open("bilat_pairs_Abers_pb2_bis.pickle", "rb") as f:
    bilat_pairs = pickle.load(f)
    

def initialize_population(init_sol, population_size):
    print(f"Génération 0: Solution initiale: {fitness(init_sol)} {init_sol}")
    population = [init_sol.copy()]
    for _ in range(1, population_size):
        new_individual = mutation(init_sol.copy(), random.randint(1, 5))
        population.append(new_individual)
    return population


def fitness(chemin) -> tuple[float, int]:
    total_distance = 0
    max_distance = 0
    index_max_distance = 0
    for i, j in itertools.islice(zip(chemin, chemin[1:]), len(chemin) - 1):
        if dist_matrix[i][j] > max_distance:
            max_distance = dist_matrix[i][j]
            index_max_distance = j

        total_distance += dist_matrix[i][j]

    return total_distance , index_max_distance


def create_dict(pairs):
    pair_dict = {}
    for pair in pairs:
        pair_dict[pair[0]] = pair[1]
        pair_dict[pair[1]] = pair[0]
    return pair_dict

bilat_pairs_dict = create_dict(bilat_pairs)

def selection(population, pool):
    fitness_values = pool.map(fitness, population)
    ranked_solutions = sorted(zip(fitness_values, population), reverse=False)
    return ranked_solutions[:POPULATION_SIZE // 6]


def calculateVariance(population):
    mean = sum(sol[0][0] for sol in population) / len(population)
    return sum((sol[0][0] - mean) ** 2 for sol in population) / len(population)


def linear_base(variance, min_variance=50000, max_variance=180000, min_base=0.0005, max_base=1.01):
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


def crossover(parent1: list[int], parent2: list[int], parent_1_worst_gene: int, parent_2_worst_gene:  int) -> list[int]:
    if parent_1_worst_gene < parent_2_worst_gene:
        segment1 = parent1[1:parent_1_worst_gene]
        segment2 = parent2[parent_1_worst_gene:parent_2_worst_gene]
        segment3 = parent1[parent_2_worst_gene:]
    else:
        segment1 = parent2[1:parent_2_worst_gene]
        segment2 = parent1[parent_2_worst_gene:parent_1_worst_gene]
        segment3 = parent2[parent_1_worst_gene:]

    child = [0]
    child_set = set(child)

    for segment in [segment1, segment2, segment3]:
        for gene in segment:
            if gene not in child_set:
                child.append(gene)
                child_set.add(gene)

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
    if random.random() < 0.2:
        subset = segment[1:-1]
        random.shuffle(subset)
        segment[1:-1] = subset
    if start-1 != 0:
        individual = swap_with_pair(individual, start -1)
    individual = swap_with_pair(individual, start)
    individual = swap_with_pair(individual, end-1)
    individual = swap_with_pair(individual, end)
    #if random.random() < 0.5:
    #    segment = segment[::-1]
    del individual[start:end]
    new_position = random.randint(1, len(individual) - 2)
    if new_position - 1 != 0:
        individual = swap_with_pair(individual, new_position - 1)
    individual = swap_with_pair(individual, new_position)
    individual = individual[:new_position] + segment + individual[new_position:]
    return individual

def swap_with_pair(individual, index=-1):
    if index == -1:
        index = random.randint(1, len(individual) - 2)
    if individual[index] in bilat_pairs_dict:
        if random.random() < 0.2:
            individual[index] = bilat_pairs_dict[individual[index]]
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
            variances.append(population_variance)

            new_population = []
            new_population.extend(individual[1] for individual in population[:population_size // 50])
            while len(new_population) < population_size // 2:
                parent1, parent2 = tournament_selection(population, population_variance), tournament_selection(
                    population, population_variance)
                child = crossover(parent1[1], parent2[1], parent1[0][1], parent2[0][1])
                mutation_length = random.randint(3, 9) if stuck_generations > 10 else random.randint(2, 6)
                if random.random() < MUTATION_RATE:
                    child = mutation(child, mutation_length)
                    random_bilat_swaps = random.randint(1, 5)
                    for _ in range(random_bilat_swaps):
                        child = swap_with_pair(child)
                new_population.append(child)

            while len(new_population) < population_size:
                parent1, parent2 = tournament_selection(population, population_variance), tournament_selection(
                    population, population_variance)
                child = crossover(parent1[1], parent2[1], parent1[0][1], parent2[0][1])
                mutation_length = random.randint(6, 18) if stuck_generations > 10 else random.randint(3, 9)
                if random.random() < MUTATION_RATE:
                    child = mutation(child, mutation_length)
                    random_bilat_swaps = random.randint(1, 8)
                    for _ in range(random_bilat_swaps):
                        child = swap_with_pair(child)
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

def ispermutation(l):
    return l[0] == 0 and l[-1] == init_solu[-1]


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
    print(f"Est-ce une bonne solution ? {ispermutation(best_solution)}")

    distance, temps = calculateDandT(init_solu)
    print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")
    print(f"fitness sol initiale : {distance + temps}")

    plt.scatter(generation, fitness)
    plt.show()