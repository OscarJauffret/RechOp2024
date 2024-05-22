import itertools
import multiprocessing
import random
import pickle
import os
import time
import math
import matplotlib.pyplot as plt
from numba import cuda, float32
import numpy as np

fitness_dtype = np.dtype([('fitness', np.float32), ('max_distance_index', np.int32)])

# Define global constants used throughout the code
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

# Change directory to load data files, handle FileNotFoundError if the path is incorrect
try:
    os.chdir("../Data/Probleme_Abers_2/")
except FileNotFoundError:
    pass

# Load necessary data from pickle files
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

def initialize_population(init_sol, population_size):
    """
    Initialize the population with the given initial solution and apply random mutations.

    Parameters:
    init_sol (np.ndarray): The initial solution from which to derive the first population.
    population_size (int): The total number of individuals in the population.

    Returns:
    np.ndarray: A 2D array representing the initial population.
    """
    population = [init_sol.copy()]
    for _ in range(1, population_size):
        new_individual = mutation(init_sol.copy(), random.randint(1, 5))
        population.append(new_individual)
    population = np.array([np.array(ind) for ind in population])
    return population


@cuda.jit
def fitness_kernel(individuals, dist_matrix, fitnesses, max_distance_indices):
    """
    CUDA kernel to calculate the fitness of multiple individuals and find the index with the maximum distance.
    
    Parameters:
    individuals (ndarray): The solutions to evaluate.
    dist_matrix (ndarray): The distance matrix.
    fitnesses (ndarray): Array to store the fitness results.
    max_distance_indices (ndarray): Array to store the index of the maximum distance for each individual.
    """
    idx = cuda.grid(1)
    n_individuals, n_cities = individuals.shape
    
    if idx < n_individuals:
        total_distance = 0.0
        max_distance = 0.0
        max_index = 0
        for i in range(n_cities - 1):
            from_city = individuals[idx, i]
            to_city = individuals[idx, i + 1]
            distance = dist_matrix[from_city, to_city]
            total_distance += distance
            if distance > max_distance:
                max_distance = distance
                max_index = i
        
        fitnesses[idx] = total_distance
        max_distance_indices[idx] = max_index


def fitness(population):
    n_individuals = population.shape[0]
    n_cities = population.shape[1]

    # Convert the population to a Numpy array
    population_np = np.array(population, dtype=np.int32)

    # Allocate memory on the device
    population_device = cuda.to_device(population_np)
    dist_matrix_device = cuda.to_device(dist_matrix)
    fitnesses_device = cuda.device_array(n_individuals, dtype=np.float32)  # Store the fitness results
    max_distance_indices_device = cuda.device_array(n_individuals, dtype=np.int32)  # Store the indices of max distances

    # Configure blocks and threads
    threads_per_block = 32
    blocks_per_grid = (n_individuals + threads_per_block - 1) // threads_per_block

    # Launch kernel
    fitness_kernel[blocks_per_grid, threads_per_block](population_device, dist_matrix_device, fitnesses_device, max_distance_indices_device)

    # Copy the results back to host
    fitnesses_result = fitnesses_device.copy_to_host()
    max_distance_indices_result = max_distance_indices_device.copy_to_host()

    # Combine the results into a list of tuples
    fitness_data = list(zip(fitnesses_result, max_distance_indices_result, population))

    # Return the fitness values and max distance indices
    return fitness_data


def create_dict(pairs):
    """
    Create a dictionary from a list of bilateral pairs for quick look-up.

    This function constructs a dictionary where each item is key-value pair representing
    bilateral links. Each pair in the list has two elements, and the function ensures that
    both elements can be used interchangeably as keys to retrieve the other element. This is
    useful for situations where you need to quickly find the corresponding pair of an element.

    Parameters:
    pairs (list of tuple): A list of tuples, where each tuple contains two elements that are bilateral.

    Returns:
    dict: A dictionary where each key is an element from the pairs, and its value is its bilateral counterpart.
    """

    pair_dict = {}
    for pair in pairs:
        # Map each element of the pair to the other
        pair_dict[pair[0]] = pair[1]
        pair_dict[pair[1]] = pair[0]
    return pair_dict



bilat_pairs_dict = create_dict(bilat_pairs)


def selection(population):
    """
    Select the top individuals from the population based on fitness.

    Parameters:
    population (np.ndarray): The current population of individuals.

    Returns:
    list: A list of tuples with the top-performing individuals and their fitness.
    """
    fitness_data = fitness(population)
    ranked_solutions = sorted(fitness_data, key=lambda x: x[0])
    return ranked_solutions[:POPULATION_SIZE // 5]

def calculateVariance(population):
    """
    Calculate the variance of fitness values within the population, used to adjust mutation parameters dynamically.

    Parameters:
    population (list): The current population whose fitness variance is to be calculated.

    Returns:
    float: The variance of the fitness values within the population.
    """
    fitness_values = [sol[0] for sol in population]
    mean = sum(fitness_values) / len(fitness_values)
    variance = sum((val - mean) ** 2 for val in fitness_values) / len(fitness_values)
    return variance


def linear_base(variance, min_variance=100000, max_variance=250000, min_base=0.0005, max_base=1.01):
    """
    Determine the mutation base rate based on the current variance of the population's fitness.

    Parameters:
    variance (float): The current variance of the population's fitness values.
    min_variance (float): The minimum threshold for variance below which exploration is encouraged.
    max_variance (float): The maximum threshold for variance above which exploitation is encouraged.
    min_base     (float): The base rate used when variance is at its minimum, promoting exploration.
    max_base (float): The base rate used when variance is at its maximum, promoting exploitation.

    Returns:
    float: The calculated mutation base rate, which influences the selection pressure during the tournament selection.
    """

    if variance <= min_variance:
        return min_base  # small base to promote exploration 
    elif variance >= max_variance:
        return max_base  # Tall base to promote exploitation. (Give higher weight to best solutions)
    else:
        return max_base + (min_base - max_base) * ((variance - min_variance) / (max_variance - min_variance))

def tournament_selection(population, variance, tournament_size=3):
    """
    Perform tournament selection from the population based on fitness calculated weights.

    Parameters:
    population (list): The current population from which to select individuals.
    variance (float): The variance of the population's fitness, affecting the selection pressure.
    tournament_size (int): The number of individuals to consider in each tournament.

    Returns:
    list: The selected individual who wins the tournament.
    """
    n = len(population)
    base = linear_base(variance)
    weights = np.exp(-base * np.arange(n))
    selected_indices = np.random.choice(n, tournament_size, p=weights/weights.sum())
    selected_individuals = [population[i] for i in selected_indices]
    return min(selected_individuals, key=lambda x: x[0])

# Cut On Worst L+R Crossover
def crossover(parent1, parent2, parent_1_worst_gene, parent_2_worst_gene):
    """
    Combine genes from two parents to create a new individual, focusing on cutting at the worst performing genes.

    Parameters:
    parent1, parent2 (np.ndarray): The parent individuals.
    parent_1_worst_gene, parent_2_worst_gene (int): Indices of the worst performing genes in each parent.

    Returns:
    np.ndarray: A new individual created by combining segments from both parents.
    """
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
                child_set.add(bilat_pairs_dict.get(gene, 0))   #if gene not in billat pair, we add 0 to the set 

    for parent in [parent1, parent2]:
        for gene in parent:
            if gene not in child_set:
                child.append(gene)
                child_set.add(gene)
                child_set.add(bilat_pairs_dict.get(gene, 0))
    return np.array(child)

def mutation(individual, length=3):
    """
    Perform a mutation on an individual by segment rearrangement and potential scrambling within the segment.

    Parameters:
    individual (np.ndarray): The individual to mutate.
    length (int): The length of the segment to extract and potentially scramble.

    Returns:
    np.ndarray: The mutated individual.
    """
    start = random.randint(1, len(individual) - 2 - length)
    end = start + length
    segment = individual[start:end].copy()

    if random.random() < SCRAMBLE_MUTATION_RATE:
        subset = segment[1:-1].copy()
        np.random.shuffle(subset)
        segment[1:-1] = subset

    if start - 1 != 0:
        individual = swap_with_pair(individual, start - 1)
    segment = swap_with_pair(segment,0)
    segment = swap_with_pair(segment, len(segment) - 1)
    individual = swap_with_pair(individual, end)

    individual = np.concatenate((individual[:start], individual[end:]))
    new_position = random.randint(1, len(individual) - 2)
    if new_position - 1 != 0:
        individual = swap_with_pair(individual, new_position - 1)
    individual = swap_with_pair(individual, new_position)
    if len(np.concatenate((individual[:new_position], segment, individual[new_position:]))) != 262:
        print(f"Taille mutaiton {len(np.concatenate((individual[:new_position], segment, individual[new_position:])))}")
        exit()
    return np.concatenate((individual[:new_position], segment, individual[new_position:]))



def swap_with_pair(individual, index=-1):
    """
    Swap a gene in the individual with its bilateral pair based on a probability.

    Parameters:
    individual (np.ndarray): The individual to mutate.
    index (int, optional): The index of the gene to potentially swap. If not specified, a random index is chosen.

    Returns:
    np.ndarray: The mutated individual.
    """

    if random.random() < 0.2:  # 20% probability to swap
        if index == -1:  # If no index is specified, choose a random one
            index = random.randint(1, len(individual) - 2)
        if individual[index] in bilat_pairs_dict:
            individual[index] = bilat_pairs_dict[individual[index]]  # Swap
    return individual

def create_new_child(population, population_variance, stuck_generations, is_heavily_mutated_part):
    """
    Create a new child by selecting parents and applying genetic operations such as crossover and mutation.

    Parameters:
    population (np.ndarray): The current population of individuals.
    population_variance (float): The variance in fitness of the current population.
    stuck_generations (int): Number of generations without significant fitness improvements.
    is_heavily_mutated_part (bool): Flag indicating if heavier mutations should be applied.

    Returns:
    np.ndarray: A new child created from selected parents with applied genetic operations.
    """

    parent1 = tournament_selection(population, population_variance)
    parent2 = tournament_selection(population, population_variance)
    child = crossover(parent1[2], parent2[2], parent1[1], parent2[1])
    if stuck_generations > 100:
        mutation_length = random.randint(10, 20)
    elif stuck_generations > 50:
        mutation_length = random.randint(5, 10)
    else:
        mutation_length = random.randint(1, 5)
    if random.random() < MUTATION_RATE:
        child = mutation(child, mutation_length)

    return child


def genetic_algorithm(init_sol, population_size, best_scores, variances):
    """
    Execute a genetic algorithm to optimize a solution using a variety of genetic operations.

    Parameters:
    init_sol (np.ndarray): The initial solution used as the baseline for generating the initial population.
    population_size (int): The size of the population in each generation.
    best_scores (list): A list to record the best fitness score in each generation.
    variances (list): A list to record the variance of fitness scores within the population for each generation.

    Returns:
    np.ndarray: The best individual (solution) found across all generations based on fitness evaluations.
    """
    population = initialize_population(init_sol, population_size)
    start_time = time.time()
    generation = 0
    stuck_generations = 0
    previous_score = 0

    while time.time() - start_time < 60:  # Run until 10 minutes
        population = selection(population)
        best_score = population[0][0]
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
        new_population.extend(individual[2] for individual in population[:population_size // 30])


        while len(new_population) < population_size // 2:
            child = create_new_child(population, population_variance, stuck_generations, False)
            new_population.append(child)

        while len(new_population) < population_size:
            child = create_new_child(population, population_variance, stuck_generations, True)
            new_population.append(child)

        #for individual in new_population:
        #    print(f"Taille des individuals {len(individual)}")

        population = np.array([np.array(ind) for ind in new_population])  # Ensure all elements are Numpy arrays

    return min(fitness(population), key=lambda x: x[0])[2]  # Return the best solution found



def calculateDandT(route):
    """
    Calculate the total distance and time for a given route.

    Parameters:
    route (np.ndarray): A route represented as an array of node indices.

    Returns:
    tuple: Total distance and time for the route.
    """
    distance = 0
    time = 0
    for i in range(len(route) - 1):
        distance += dist_matrix[route[i], route[i + 1]]
        time += dur_matrix[route[i], route[i + 1]]
        time += collection_time[route[i]]
    time += collection_time[route[-1]]
    return distance, time


def has_duplicates(lst):
    """
    Check if a list contains duplicate elements.

    Parameters:
    lst (list): The list to check for duplicates.

    Returns:
    bool: True if there are duplicates, otherwise False.
    """

    return len(lst) != len(set(lst))

def ispermutation(l):
    """
    Verify if a list is a valid permutation of the expected node indices.

    Parameters:
    l (list): The list representing a route.

    Returns:
    bool: True if the list is a permutation of the range from 0 to the last index in the initial solution, otherwise False.
    """

    return l[0] == 0 and l[-1] == init_solu[-1]


if __name__ == "__main__":
    best_scores = []
    variances = []
    best_solution = genetic_algorithm(np.array(init_solu), POPULATION_SIZE, best_scores, variances)

    generations = np.arange(len(best_scores))
    fitness_scores = np.array(best_scores)

    print(f"La meilleure solution jamais obtenue est : {min(best_scores)}")

    print(best_solution)
    distance, temps = calculateDandT(best_solution)
    print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")
    print(f"Fitness: {distance + temps}")
    print(has_duplicates(best_solution.tolist()))
    print(f"Est-ce une bonne solution ? {ispermutation(best_solution.tolist())}")

    distance, temps = calculateDandT(np.array(init_solu))
    print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")
    print(f"Fitness de la solution initiale : {distance + temps}")

    plt.scatter(generations, fitness_scores)
    plt.xlabel('Génération')
    plt.ylabel('Fitness')
    plt.title('Évolution de la fitness au cours des générations')
    plt.show()

