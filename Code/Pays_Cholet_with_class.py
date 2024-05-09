import random
import pickle
import time
from matplotlib import pyplot as plt
import math
from Individual import Individual

ACCELERATED_MUTATION_THRESHOLD = 1000
POPULATION_SIZE = 500
WEIGHT_LIMIT = 5850
bad = 999999
ACCELERATED_MUTATION_NUMBER = 3
OLD_GENERATION = 200

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


def initialize_population(init_sol: list[int], population_size: int) -> list[Individual]:
    population = [Individual(init_sol.copy()) for _ in range(population_size)]
    for individual in population:
        subset = individual.get_chromosome()[1:-1]
        random.shuffle(subset)
        # subset = mutation(subset)
        individual.get_chromosome()[1:-1] = subset

    return population


def selection(population: list[Individual]) -> list[Individual]:
    ranked_individuals = sorted([individual for individual in population], key=lambda x: x.get_fitness())
    return ranked_individuals[:POPULATION_SIZE // 10]


def calculate_variance(population: list[Individual]) -> float:
    mean = sum(sol.get_fitness() for sol in population) / len(population)
    return sum((sol.get_fitness() - mean) ** 2 for sol in population) / len(population)


def linear_base(variance: float, min_variance: int = 100000, max_variance: int = 1000000, min_base: float = 0.0005,
                max_base: float = 1.01) -> float:
    if variance <= min_variance:
        return max_base  # Grande base pour favoriser l'exploitation
    elif variance >= max_variance:
        return min_base  # Petite base pour favoriser l'exploration
    else:
        return max_base + (min_base - max_base) * ((variance - min_variance) / (max_variance - min_variance))


def tournament_selection(population: list[Individual], variance: float, tournament_size: int = 10) -> Individual:
    n = len(population)
    # weights = [n - i for i in range(n)]
    base = linear_base(variance)  # exponential_base(variance)
    weights = [math.exp(-base * i) for i in range(n)]
    # weights[1:] = [w*2 for w in weights[1:]]
    # if variance > 230000:
    #    weights = [math.exp(-0.4 * i) for i in range(n)]
    # else:
    #    weights = [pow(1.05, -i) for i in range(n)]
    tournament = random.choices(population, weights=weights, k=tournament_size)
    tournament = sorted(tournament, key=lambda x: x.get_fitness())

    p = 0.9
    tournament_weights = [p * ((1 - p) ** i) for i in range(tournament_size)]
    return random.choices(tournament, weights=tournament_weights, k=1)[0]


def crossover(parent1: Individual, parent2: Individual) -> Individual:
    child = Individual([0])
    for j in range(1, len(parent1.get_chromosome()) // 2):
        child.get_chromosome().append(parent1.get_chromosome()[j])
    for element in parent2.get_chromosome():
        if element not in child.get_chromosome():
            child.get_chromosome().append(element)
    return child


def mutation(individual: Individual, length: int = 3) -> Individual:
    start = random.randint(1, len(individual.get_chromosome()) - 2 - length)
    end = start + length
    segment = individual.get_chromosome()[start:end]
    del individual.get_chromosome()[start:end]
    new_position = random.randint(1, len(individual.get_chromosome()) - 2)
    individual.get_chromosome()[:] = individual.get_chromosome()[:new_position] + segment + individual.get_chromosome()[new_position:]
    return individual


def genetic_algorithm(init_sol, population_size, best_scores, carried_over_proportion=0.2) -> list[int]:
    population = initialize_population(init_sol, population_size)
    start_time = time.time()
    generation = 0
    while time.time() - start_time < 600:
        population = selection(population)

        best_score = population[0].get_fitness()
        best_scores.append(best_score)

        generation += 1
        print(f"Generation {generation}: {best_score}", end=" ")

        population_variance = calculate_variance(population)
        print(f"Variance: {population_variance}")

        new_population = []

        new_population.extend(individual for individual in
                              population[:int(round(len(population) * carried_over_proportion, 0))])

        while len(new_population) < population_size // 2:
            parent1 = tournament_selection(population, population_variance)
            parent2 = tournament_selection(population, population_variance)

            child = crossover(parent1, parent2)
            child = mutation(child, random.randint(1, 5))
            new_population.append(child)

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, population_variance)
            parent2 = tournament_selection(population, population_variance)

            child = crossover(parent1, parent2)
            child = mutation(child, random.randint(10, 15))
            new_population.append(child)

        population = new_population

    return selection(population)[0].get_chromosome()


def calculate_d_and_t(l):
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
best_solution = genetic_algorithm(init_solu, POPULATION_SIZE, best_scores)

generation = [i for i in range(len(best_scores))]
fitness = [s for s in best_scores]

print(f"La meilleure solutions jamais obtenue est : {min(best_scores)}")

print(best_solution)
distance, temps = calculate_d_and_t(best_solution)
print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")
print(f"Fitness: {distance + temps}")
print(has_duplicates(best_solution))

distance, temps = calculate_d_and_t(init_solu)
print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")
print(f"fitness sol initiale : {distance + temps}")

plt.scatter(generation, fitness)
plt.show()
