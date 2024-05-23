import itertools
import multiprocessing
import random
import pickle
import os
import time
import math
import matplotlib.pyplot as plt
from numba import cuda, int32
import numpy as np

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

def initialize_population_gpu(init_sol, population_size, mutation_rate=0.9):
    n_cities = len(init_sol)
    # Créer un tableau sur le GPU où la population sera stockée
    population_device = cuda.device_array((population_size, n_cities), dtype=np.int32)

    # Initialiser la première solution
    population_device[0] = init_sol

    # Dupliquer la solution initiale pour remplir la population
    for i in range(1, population_size):
        population_device[i] = init_sol

    # Configurer les états de génération aléatoire pour chaque thread
    rng_states = cuda.random.create_xoroshiro128p_states(population_size * n_cities, seed=12)

    # Appliquer la mutation à la population
    threads_per_block = 256
    blocks_per_grid = (population_size + threads_per_block - 1) // threads_per_block
    mutation_kernel[blocks_per_grid, threads_per_block](population_device, mutation_rate, n_cities, rng_states)

    return population_device


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

@cuda.jit
def tournament_selection_kernel(fitnesses, indices, winners, tournament_size):
    # Chaque thread gère un tournoi
    idx = cuda.grid(1)
    if idx < winners.size:
        best_index = -1
        best_fitness = float('inf')
        for i in range(tournament_size):
            competitor_idx = indices[idx * tournament_size + i]
            competitor_fitness = fitnesses[competitor_idx]
            if competitor_fitness < best_fitness:
                best_fitness = competitor_fitness
                best_index = competitor_idx
        winners[idx] = best_index

def run_tournament_selection_on_gpu(fitnesses, population_size, tournament_size):
    n_tournaments = population_size // tournament_size

    # Créer un tableau de tous les indices sélectionnés pour les tournois
    all_indices = np.random.randint(0, fitnesses.size, size=n_tournaments * tournament_size)
    indices_device = cuda.to_device(all_indices)
    winners_device = cuda.device_array(n_tournaments, dtype=np.int32)

    # Configuration du kernel
    threads_per_block = 256
    blocks_per_grid = (n_tournaments + threads_per_block - 1) // threads_per_block

    # Exécution du kernel
    tournament_selection_kernel[blocks_per_grid, threads_per_block](fitnesses, indices_device, winners_device, tournament_size)

    # Récupérer les indices des gagnants
    winners = winners_device.copy_to_host()
    return winners

@cuda.jit
def crossover_kernel(parents, children, parent_indices, n_cities):
    idx = cuda.grid(1)
    state = cuda.random.create_xoroshiro128p_states(cuda.gridsize(1), seed=42)
    if idx < parent_indices.size // 2:
        parent1_idx = parent_indices[2 * idx]
        parent2_idx = parent_indices[2 * idx + 1]

        # Sélection aléatoire de deux points pour le crossover d'ordre
        point1 = int(cuda.random.xoroshiro128p_uniform_float32(state, idx) * n_cities)
        point2 = int(cuda.random.xoroshiro128p_uniform_float32(state, idx) * n_cities)
        
        if point1 > point2:
            point1, point2 = point2, point1

        # Initialiser le tableau d'enfant avec des valeurs invalides
        for i in range(n_cities):
            children[2 * idx][i] = -1
            children[2 * idx + 1][i] = -1

        # Copie de la section du parent 1 à l'enfant 1
        for i in range(point1, point2 + 1):
            children[2 * idx][i] = parents[parent1_idx][i]

        # Remplissage de l'enfant 1 avec les villes du parent 2 en partant de point2 + 1
        fill_idx = (point2 + 1) % n_cities
        for i in range(n_cities):
            city = parents[parent2_idx][(point2 + 1 + i) % n_cities]
            if city not in children[2 * idx]:
                if fill_idx >= n_cities:
                    fill_idx = 0
                children[2 * idx][fill_idx] = city
                fill_idx += 1

def crossover_gpu(parents_device, n_individuals, n_cities):
    # Allocation de l'espace pour les enfants sur le GPU
    children_device = cuda.device_array((n_individuals, n_cities), dtype=np.int32)

    # Générer des indices de parents aléatoires pour le croisement
    parent_indices = np.random.permutation(n_individuals)
    parent_indices_device = cuda.to_device(parent_indices)

    # Configuration du kernel
    threads_per_block = 256
    blocks_per_grid = (n_individuals // 2 + threads_per_block - 1) // threads_per_block

    # Appel du kernel
    crossover_kernel[blocks_per_grid, threads_per_block](parents_device, children_device, parent_indices_device, n_cities)

    # Retourne le tableau des enfants
    return children_device


@cuda.jit
def mutation_kernel(individuals, mutation_rate, segment_length, n_cities):
    idx = cuda.grid(1)
    state = cuda.random.create_xoroshiro128p_states(cuda.gridsize(1), seed=42)
    
    if idx < individuals.shape[0] and cuda.random.xoroshiro128p_uniform_float32(state, idx) < mutation_rate:
        # Sélectionner un point de départ aléatoire pour le segment
        start = int(cuda.random.xoroshiro128p_uniform_float32(state, idx) * (n_cities - segment_length))
        end = start + segment_length

        # Extraire le segment et mélanger potentiellement
        segment = cuda.local.array(20, dtype=int32)  # Assumer la longueur maximale de 20 pour le segment
        for i in range(segment_length):
            segment[i] = individuals[idx][start + i]

        # Mélange simple du segment
        for i in range(1, segment_length-1):  # Ne mélangez pas le premier et le dernier élément
            j = int(cuda.random.xoroshiro128p_uniform_float32(state, idx) * (segment_length - 2)) + 1
            temp = segment[i]
            segment[i] = segment[j]
            segment[j] = temp

        # Réinsérer le segment à une nouvelle position
        new_position = int(cuda.random.xoroshiro128p_uniform_float32(state, idx) * (n_cities - segment_length))
        # Décaler le reste pour faire de la place au segment
        if new_position < start:
            for i in range(start - new_position):
                individuals[idx][end - 1 - i] = individuals[idx][start - 1 - i]
        elif new_position > start:
            for i in range(end - start):
                individuals[idx][new_position + i] = segment[i]

def mutation_gpu(individuals_device, mutation_rate, segment_length, n_individuals, n_cities):
    threads_per_block = 256
    blocks_per_grid = (n_individuals + threads_per_block - 1) // threads_per_block

    mutation_kernel[blocks_per_grid, threads_per_block](individuals_device, mutation_rate, segment_length, n_cities)

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

def select_parents_on_gpu(fitnesses_device, population_size):
    # Assuming a function exists to perform tournament selection on GPU and returns indices
    tournament_size = 3  # Or any other logic you have for determining the size
    return run_tournament_selection_on_gpu(fitnesses_device, population_size, tournament_size)

def determine_mutation_length(stuck_generations):
    if stuck_generations > 100:
        return np.random.randint(10, 20)
    elif stuck_generations > 50:
        return np.random.randint(5, 10)
    else:
        return np.random.randint(1, 5)


def create_new_child_gpu(population_device, fitnesses_device, population_size, n_cities, stuck_generations, mutation_rate):
    """
    Create a new child by selecting parents and applying genetic operations such as crossover and mutation on GPU.

    Parameters:
    population_device (DeviceNDArray): The current population of individuals on the GPU.
    fitnesses_device (DeviceNDArray): The fitness scores of the current population on the GPU.
    population_size (int): The size of the population.
    n_cities (int): Number of cities in the TSP.
    stuck_generations (int): Number of generations without significant fitness improvements.
    mutation_rate (float): The probability of mutation.

    Returns:
    DeviceNDArray: The new child created from selected parents with applied genetic operations.
    """

    # Select parents on GPU
    parent_indices_device = select_parents_on_gpu(fitnesses_device, population_size)

    # Crossover to create a new child
    children_device = crossover_gpu(population_device, parent_indices_device, n_cities)

    # Determine mutation length based on the number of stuck generations
    mutation_length = determine_mutation_length(stuck_generations)

    # Apply mutation on GPU
    mutation_gpu(children_device, mutation_rate, mutation_length, population_size, n_cities)

    return children_device

def create_new_generation_gpu(population_device, fitnesses_device, population_size, n_cities, stuck_generations, mutation_rate):
    """
    Generate a new population on the GPU by creating new children using genetic operations.
    
    Parameters:
    population_device (DeviceNDArray): The current population stored on the GPU.
    fitnesses_device (DeviceNDArray): The fitness values for the current population stored on the GPU.
    population_size (int): The size of the population.
    n_cities (int): The number of cities in the TSP problem.
    stuck_generations (int): The number of generations without significant fitness improvements.
    mutation_rate (float): The mutation rate to be used during mutation.
    
    Returns:
    DeviceNDArray: The new population generated on the GPU.
    """
    # Allocate space for the new population on the GPU
    new_population_device = cuda.device_array((population_size, n_cities), dtype=np.int32)

    # Create a random generator states for each thread
    rng_states = cuda.random.create_xoroshiro128p_states(population_size * n_cities, seed=42)

    # Kernel dimensions
    threads_per_block = 256
    blocks_per_grid = (population_size + threads_per_block - 1) // threads_per_block

    # Generate each new child on the GPU
    for i in range(population_size):
        create_new_child_gpu[blocks_per_grid, threads_per_block](
            population_device, fitnesses_device, new_population_device, i, n_cities, stuck_generations, mutation_rate, rng_states
        )

    return new_population_device

def genetic_algorithm_gpu(init_sol, population_size, best_scores, variances, n_cities):
    """
    Execute a genetic algorithm entirely on the GPU to optimize a TSP solution.

    Parameters:
    init_sol (np.ndarray): The initial solution used as the baseline for generating the initial population.
    population_size (int): The size of the population in each generation.
    best_scores (list): A list to record the best fitness score in each generation.
    variances (list): A list to record the variance of fitness scores within the population for each generation.
    n_cities (int): The number of cities in the problem, required for array dimensions.

    Returns:
    np.ndarray: The best individual (solution) found across all generations based on fitness evaluations.
    """
    # Initialize population on GPU
    population_device = initialize_population_gpu(init_sol, population_size, n_cities)

    # Allocate space for fitnesses on GPU
    fitnesses_device = cuda.device_array(population_size, dtype=np.float32)

    start_time = time.time()
    generation = 0
    stuck_generations = 0
    previous_score = float('inf')

    while time.time() - start_time < 600:  # Run until 10 minutes
        # Calculate fitnesses
        fitness(population_device, fitnesses_device)

        # Get the best score for recording and decision making
        best_score = cuda.to_host(fitnesses_device).min()
        best_scores.append(best_score)

        # Determine population variance
        population_variance = calculateVariance(fitnesses_device)
        variances.append(population_variance)

        # Print progress
        print(f"Generation {generation}: {best_score} with Variance: {population_variance}")

        if best_score == previous_score:
            stuck_generations += 1
        else:
            stuck_generations = 0
            previous_score = best_score

        # Create new population
        new_population_device = create_new_generation_gpu(population_device, fitnesses_device, population_size, n_cities, stuck_generations)

        # Swap populations
        population_device = new_population_device
        generation += 1

    # Extract the best individual
    return extract_best_individual_gpu(population_device, fitnesses_device)


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

