import itertools
import multiprocessing
import random
import pickle
import os
import time
import math
import matplotlib.pyplot as plt


ACCELERATED_MUTATION_THRESHOLD = 1000
POPULATION_SIZE = 1000
WEIGHT_LIMIT = 5850
bad = 999999
ACCELERATED_MUTATION_NUMBER = 3
OLD_GENERATION = 200
MUTATION_RATE = 0.9
SCRAMBLE_MUTATION_RATE = 0.2
NORMAL_MUTATION_LENGTHS = [(2, 6), (3, 9)]  # [(random.randint(2, 5), random.randint(6, 8)), (random.randint(3, 6), random.randint(8, 10))]
HEAVY_MUTATION_LENGTHS = [(3, 9), (6, 18)]  # [(random.randint(3, 6), random.randint(8, 10)), (random.randint(6, 10), random.randint(15, 20))]
try:
    os.chdir("../Data/Probleme_Abers_2/")
except FileNotFoundError:
    pass
# Load necessary data
with open("init_sol_Abers_pb2.pickle", "rb") as f:
    init_solu = pickle.load(f)

with open("dist_matrix_Abers_pb2.pickle", "rb") as f:
    dist_matrix = pickle.load(f)

with open("weight_Abers_pb2.pickle", "rb") as f:
    weight_list = pickle.load(f)

with open("dur_matrix_Abers_pb2.pickle", "rb") as f:
    dur_matrix = pickle.load(f)

with open("temps_collecte_Abers_pb2.pickle", "rb") as f:
    collection_time = pickle.load(f)

with open("bilat_pairs_Abers_pb2.pickle", "rb") as f:
    bilat_pairs = pickle.load(f)


# init_solu = [0, 181, 305, 34, 272, 35, 47, 151, 155, 31, 32, 232, 330, 331, 231, 275, 71, 279, 72, 303, 73, 25, 161, 109, 283, 298, 306, 230, 83, 280, 307, 1, 299, 346, 26, 27, 166, 229, 300, 228, 302, 149, 58, 289, 292, 148, 107, 17, 24, 28, 293, 152, 138, 59, 332, 60, 61, 282, 222, 226, 335, 328, 136, 56, 100, 53, 278, 68, 179, 159, 150, 122, 309, 88, 273, 127, 106, 13, 295, 14, 15, 177, 21, 22, 23, 294, 147, 99, 115, 271, 57, 98, 125, 126, 204, 40, 96, 97, 79, 205, 291, 327, 326, 203, 315, 202, 313, 201, 274, 241, 233, 286, 316, 101, 314, 102, 270, 158, 277, 37, 38, 39, 41, 281, 42, 16, 290, 199, 62, 63, 64, 105, 135, 322, 312, 266, 252, 49, 66, 78, 262, 255, 55, 156, 2, 139, 33, 319, 163, 162, 251, 141, 29, 30, 154, 343, 341, 342, 153, 140, 8, 256, 324, 257, 258, 43, 44, 45, 46, 84, 304, 85, 297, 86, 137, 70, 146, 145, 160, 311, 310, 287, 267, 338, 268, 339, 253, 260, 48, 261, 133, 334, 134, 254, 269, 263, 333, 164, 189, 74, 75, 76, 329, 340, 124, 264, 265, 111, 108, 318, 113, 130, 131, 132, 276, 94, 296, 95, 284, 224, 321, 235, 3, 4, 5, 9, 10, 11, 12, 80, 144, 128, 345, 143, 142, 301, 6, 7, 337, 259, 18, 19, 103, 323, 104, 157, 285, 36, 320, 288, 194, 308, 193, 20, 317, 325, 336, 50, 51, 344, 52, 347]


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

    return total_distance, index_max_distance


def create_dict(pairs):
    pair_dict = {}
    for pair in pairs:
        pair_dict[pair[0]] = pair[1]
        pair_dict[pair[1]] = pair[0]
    return pair_dict


bilat_pairs_dict = create_dict(bilat_pairs)


def selection(population, pool, stuck_generations):
    fitness_values = pool.map(fitness, population)
    ranked_solutions = sorted(zip(fitness_values, population), reverse=False)
    return ranked_solutions[:POPULATION_SIZE // 5]


def calculateVariance(population):
    mean = sum(sol[0][0] for sol in population) / len(population)
    return sum((sol[0][0] - mean) ** 2 for sol in population) / len(population)


# NB : le 41.8km a été obtenu en utilisant max_variance = 1_000_000 et min_variance = 100_000
def linear_base(variance, min_variance=100000, max_variance=250000, min_base=0.0005, max_base=1.01):
    if variance <= min_variance:
        return min_base  # Petite base pour favoriser l'exploration
    elif variance >= max_variance:
        return max_base  # Grande base pour favoriser l'exploitation, donc on donne plus de poids aux meilleures solutions
    else:
        return max_base + (min_base - max_base) * ((variance - min_variance) / (max_variance - min_variance))


def tournament_selection(population, variance, tournament_size=3):
    n = len(population)
    base = linear_base(variance)
    weights = [math.exp(-base * i) for i in range(n)]
    return min(random.choices(population, weights=weights, k=tournament_size))

def crossover(parent1: list[int], parent2: list[int], parent_1_worst_gene: int, parent_2_worst_gene: int) -> list[int]:
    # Définir les segments en coupant les parents sur leurs pires gènes
    if parent_1_worst_gene < parent_2_worst_gene:
        segment1 = parent1[1:parent_1_worst_gene]
        segment2 = parent2[parent_1_worst_gene:parent_2_worst_gene]
        segment3 = parent1[parent_2_worst_gene:]
    else:
        segment1 = parent2[1:parent_2_worst_gene]
        segment2 = parent1[parent_2_worst_gene:parent_1_worst_gene]
        segment3 = parent2[parent_1_worst_gene:]

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
    if random.random() < SCRAMBLE_MUTATION_RATE:
        subset = segment[1:-1]
        random.shuffle(subset)
        segment[1:-1] = subset
        #index = random.randint(1, len(segment) - 2)
        #segment[index], segment[index + 1] = segment[index + 1], segment[index]
    if start - 1 != 0:
        individual = swap_with_pair(individual, start - 1)
    individual = swap_with_pair(individual, start)
    individual = swap_with_pair(individual, end - 1)
    individual = swap_with_pair(individual, end)
    del individual[start:end]
    new_position = random.randint(1, len(individual) - 2)
    if new_position - 1 != 0:
        individual = swap_with_pair(individual, new_position - 1)
    individual = swap_with_pair(individual, new_position)
    individual = individual[:new_position] + segment + individual[new_position:]
    return individual


def swap_with_pair(individual, index=-1):
    if random.random() < 0.2:
        if index == -1:
            index = random.randint(1, len(individual) - 2)
        if individual[index] in bilat_pairs_dict:
            individual[index] = bilat_pairs_dict[individual[index]]
    return individual


def create_new_child(population, population_variance, stuck_generations, is_heavily_mutated_part):
    if is_heavily_mutated_part:
        # Format: [(Normal mutation min length, Normal mutation max length), (Accelerated mutation min length, Accelerated mutation max length)]
        mutation_length_values = HEAVY_MUTATION_LENGTHS
        # Format: (Random bilateral swaps min, Random bilateral swaps max)
        random_bilat_swaps_values = (1, 2)
    else:
        mutation_length_values = NORMAL_MUTATION_LENGTHS
        random_bilat_swaps_values = (1, 3)
    parent1 = tournament_selection(population, population_variance)
    parent2 = tournament_selection(population, population_variance)
    child = crossover(parent1[1], parent2[1], parent1[0][1], parent2[0][1])
    if stuck_generations > 100:
        mutation_length = random.randint(10, 20)
    elif stuck_generations > 50:
        mutation_length = random.randint(5, 10)
    else:
        mutation_length = random.randint(1, 5)
    #mutation_length = random.randint(mutation_length_values[1][0], mutation_length_values[1][1]) if stuck_generations > 10 else random.randint(mutation_length_values[0][0], mutation_length_values[0][1])
    if random.random() < MUTATION_RATE:
        child = mutation(child, mutation_length)
        #random_bilat_swaps = random.randint(random_bilat_swaps_values[0], random_bilat_swaps_values[1])
        #for _ in range(random_bilat_swaps):
        #    child = swap_mutation(child)

    return child

def swap_mutation(individual):
    first_index = random.randint(1, len(individual) - 2)
    second_index = random.randint(1, len(individual) - 2)
    individual[first_index], individual[second_index] = individual[second_index], individual[first_index]
    return individual


def genetic_algorithm(init_sol, population_size, best_scores, variances):
    population = initialize_population(init_sol, population_size)
    start_time = time.time()
    generation = 0
    stuck_generations = 0
    previous_score = 0
    with multiprocessing.Pool() as pool:
        while time.time() - start_time < 600:
            population = selection(population, pool, stuck_generations)
            # print(f"Genetic diversity: {genetic_diversity(population)}")

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
            new_population.extend(individual[1] for individual in population[:population_size // 30])
            while len(new_population) < population_size // 2:
                child = create_new_child(population, population_variance, stuck_generations, False)
                new_population.append(child)

            while len(new_population) < population_size:
                child = create_new_child(population, population_variance, stuck_generations, True)
                new_population.append(child)

            #while len(new_population) < population_size:
            #    new_individual = mutation(init_solu.copy(), random.randint(1, 5))
            #    new_population.append(new_individual)

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

#if __name__ == "__main__":
#    best_scores = []
#    variances = []
#
#    best_solution = genetic_algorithm(init_solu, POPULATION_SIZE, best_scores, variances)
#
#    result = {
#        "best_score": min(best_scores),
#        "best_solution": best_solution,
#        "distance_time": calculateDandT(best_solution),
#        "generation_best_scores": best_scores,
#        "scramble_mutation_rate": SCRAMBLE_MUTATION_RATE,
#        "normal_mutation_lengths": NORMAL_MUTATION_LENGTHS,
#        "heavy_mutation_lengths": HEAVY_MUTATION_LENGTHS,
#    }
#
#    os.chdir("../../Code")
#    with open("result.pkl", "wb") as f:
#        pickle.dump(result, f)