import itertools
import multiprocessing
import random
import pickle
import os
import time
import math
import matplotlib.pyplot as plt
from utils import *

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
init_solu = load_data("init_sol_Abers_pb2.pickle")
dist_matrix = load_data("dist_matrix_Abers_pb2.pickle")
weight_list = load_data("weight_Abers_pb2.pickle")
dur_matrix = load_data("dur_matrix_Abers_pb2.pickle")
collection_time = load_data("temps_collecte_Abers_pb2.pickle")
bilat_pairs = load_data("bilat_pairs_Abers_pb2.pickle")

bilat_pairs_dict = create_dict(bilat_pairs)

def initialize_population(init_sol, population_size):
    """
    Initialize the population with the given initial solution and apply random mutations.

    Parameters:
    init_sol (list): The initial solution from which to derive the first population.
    population_size (int): The total number of individuals in the population.

    Returns:
    list: A list of individuals representing the initial population.
    """

    #print(f"Génération 0: Solution initiale: {fitness(init_sol)} {init_sol}")
    population = [init_sol.copy()]
    for _ in range(1, population_size):
        new_individual = mutation(init_sol.copy(), random.randint(1, 5))
        population.append(new_individual)
    return population



def fitness(chemin) -> tuple[float, int]:
    """
    Calculate the fitness of a solution.

    The fitness is calculated as the total distance of the solution. 
    The function also finds the maximum distance between any two points in the solution and its index.

    Parameters:
    chemin (list): The solution to evaluate.

    Returns:
    tuple: The total distance and the index of the maximum distance.
    """

    total_distance = 0
    max_distance = 0
    index_max_distance = 0
    for i, j in itertools.islice(zip(chemin, chemin[1:]), len(chemin) - 1):
        if dist_matrix[i][j] > max_distance:
            max_distance = dist_matrix[i][j]
            index_max_distance = j
        total_distance += dist_matrix[i][j]

    return total_distance, index_max_distance



def selection(population, pool):
    """
    Select the top individuals from the population based on fitness calculated in parallel.

    Parameters:
    population (list): The current population of individuals.
    pool (multiprocessing.Pool): A pool of worker processes to parallelize fitness calculations.

    Returns:
    list: A subset of the population consisting of the top-performing individuals.
    """

    fitness_values = pool.map(fitness, population)
    ranked_solutions = sorted(zip(fitness_values, population), reverse=False)
    return ranked_solutions[:POPULATION_SIZE // 5]


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
    weights = [math.exp(-base * i) for i in range(n)]
    return min(random.choices(population, weights=weights, k=tournament_size))

# Cut On Worst L+R Crossover
def crossover(parent1: list[int], parent2: list[int], parent_1_worst_gene: int, parent_2_worst_gene: int) -> list[int]:
    """
    Combine genes from two parents to create a new individual, focusing on cutting at the worst performing genes.

    Parameters:
    parent1, parent2 (list): The parent individuals.
    parent_1_worst_gene, parent_2_worst_gene (int): Indices of the worst performing genes in each parent.

    Returns:
    list: A new individual created by combining segments from both parents.
    """
        
    # Define segments by cutting parents on their worst genes
    if parent_1_worst_gene < parent_2_worst_gene:
        segment1 = parent1[1:parent_1_worst_gene]
        segment2 = parent2[parent_1_worst_gene:parent_2_worst_gene]
        segment3 = parent1[parent_2_worst_gene:]
    else:
        segment1 = parent2[1:parent_2_worst_gene]
        segment2 = parent1[parent_2_worst_gene:parent_1_worst_gene]
        segment3 = parent2[parent_1_worst_gene:]

    # Initialize the child with the first common element
    child = [0]
    child_set = set(child)

    # Add segments to the child while avoiding duplicates
    for segment in [segment1, segment2, segment3]:
        for gene in segment:
            if gene not in child_set:
                child.append(gene)
                child_set.add(gene)
                child_set.add(bilat_pairs_dict.get(gene, 0)) 

    # Add remaining elements from parents to complete the child
    for parent in [parent1, parent2]:
        for gene in parent:
            if gene not in child_set:
                child.append(gene)
                child_set.add(gene)
                child_set.add(bilat_pairs_dict.get(gene, 0)) 
    return child


def mutation(individual, length=3):
    """
    Perform a mutation on an individual by rearranging a segment and potentially scrambling it, 
    followed by bilateral pair swaps.

    This function selects a segment based on the given length, scrambles the contents if a random 
    condition based on SCRAMBLE_MUTATION_RATE is met, and then reinserts the segment at a new 
    position. It may swap adjacent genes with their bilateral pairs before and after the segment.

    Parameters:
    individual (list): The individual to mutate, represented as a list of gene indices.
    length (int): The length of the segment to extract and potentially scramble.

    Returns:
    list: The mutated individual with the segment rearranged and adjacent genes potentially swapped.
    """
    # Randomly select the starting point of the segment to mutate
    start = random.randint(1, len(individual) - 2 - length)
    end = start + length
    segment = individual[start:end]  # Extract the segment to be potentially scrambled

    # Optionally scramble the segment with a certain probability
    if random.random() < SCRAMBLE_MUTATION_RATE:
        subset = segment[1:-1]  # Exclude the first and last element from scrambling
        random.shuffle(subset)  # Scramble the subset
        segment[1:-1] = subset  # Replace the original segment content with the scrambled subset

    # Swap adjacent genes with their pairs before and after the segment if they exist
    if start - 1 != 0:
        individual = swap_with_pair(individual, start - 1)  # Swap with pair at the beginning of the segment
    individual = swap_with_pair(individual, start)  # Swap the start of the segment
    individual = swap_with_pair(individual, end - 1)  # Swap with pair at the end of the segment
    individual = swap_with_pair(individual, end)  # Swap the end of the segment

    # Remove the original segment
    del individual[start:end]

    # Randomly select a new position to reinsert the segment
    new_position = random.randint(1, len(individual) - 2)
    if new_position - 1 != 0:
        individual = swap_with_pair(individual, new_position - 1)  # Swap with pair before the new position if any
    individual = swap_with_pair(individual, new_position)  # Swap the new position

    # Reinsert the segment at the new position
    individual = individual[:new_position] + segment + individual[new_position:]
    return individual


def swap_with_pair(individual, index=-1):
    """
    Swap a gene in the individual with its bilateral pair based on a probability.

    This function modifies an individual by potentially swapping one of its genes with its corresponding bilateral pair.
    The swap occurs with a 20% chance and can target a specific index or a random index if not specified.

    Parameters:
    individual (list): The individual to mutate.
    index (int, optional): The index of the gene to potentially swap. If not specified, a random index is chosen.

    Returns:
    list: The mutated individual with the gene possibly swapped.
    """

    if random.random() < 0.2:  # 20% probability to swap
        if index == -1:  # If no index is specified, choose a random one
            index = random.randint(1, len(individual) - 2)
        if individual[index] in bilat_pairs_dict:
            individual[index] = bilat_pairs_dict[individual[index]]  # Swap
    return individual

def create_new_child(population, population_variance, stuck_generations):
    """
    Create a new child by selecting parents and applying genetic operations based on the population's current state.

    Parameters:
    population (list): The current population of individuals.
    population_variance (float): The variance in fitness within the current population, affecting mutation strategies.
    stuck_generations (int): The number of consecutive generations without significant fitness improvements.
    is_heavily_mutated_part (bool): Indicates whether to apply more aggressive mutations.

    Returns:
    list: A new child created through genetic operations including crossover and mutation.
    """

    parent1 = tournament_selection(population, population_variance)
    parent2 = tournament_selection(population, population_variance)
    child = crossover(parent1[1], parent2[1], parent1[0][1], parent2[0][1])
    if stuck_generations > 100:
        mutation_length = random.randint(10, 20)
    elif stuck_generations > 50:
        mutation_length = random.randint(5, 10)
    else:
        mutation_length = random.randint(1, 5)
    if random.random() < MUTATION_RATE:
        child = mutation(child, mutation_length)

    return child


def genetic_algorithm(init_sol, population_size, best_scores):
    """
    Execute a genetic algorithm to optimize a solution using a variety of genetic operations.

    This function iteratively improves a population of solutions over several generations using selection,
    crossover, and mutation. It tracks the best scores and variances of the population to dynamically adjust
    genetic operations and stop when a time limit is reached.

    Parameters:
    init_sol (list): The initial solution used as the baseline for generating the initial population.
    population_size (int): The size of the population in each generation.
    best_scores (list): A list to record the best fitness score in each generation.
    variances (list): A list to record the variance of fitness scores within the population for each generation.

    Returns:
    list: The best individual (solution) found across all generations based on fitness evaluations.
    """

    population = initialize_population(init_sol, population_size)   # Initialize the population
    start_time = time.time()
    generation = 0
    stuck_generations = 0
    previous_score = 0

    with multiprocessing.Pool() as pool:     # Use multiprocessing to handle parallel computation of fitness evaluations
        while time.time() - start_time < 600:    # Run until 10 minutes
            
            population = selection(population, pool)
            # print(f"Genetic diversity: {genetic_diversity(population)}")
            best_score = population[0][0][0]
            best_scores.append(best_score)

            # Check if the best score has changed from the previous generation
            if best_score == previous_score:
                stuck_generations += 1
            else:
                stuck_generations = 0
                previous_score = best_score
            generation += 1
            print(f"Generation {generation}: {best_score}", end=" ")
            
            # Calculate the variance of fitness scores in the population for dynamic adaptations
            population_variance = calculate_variance(population)
            print(f"Variance: {population_variance}")

            # Create a new population starting with the best performing individuals
            new_population = []
            new_population.extend(individual[1] for individual in population[:population_size // 30])
            
            # Fill a part of the new population with offspring from selected parents
            while len(new_population) < population_size // 2:   #1000/2 = 500 child form best parents choose with tournament selection
                child = create_new_child(population, population_variance, stuck_generations)
                new_population.append(child)

            # Continue populating until the desired population size is reached
            while len(new_population) < population_size:
                child = create_new_child(population, population_variance, stuck_generations)
                new_population.append(child)

            population = new_population # Replace the old population with the new one

    return min(population, key=fitness)



if __name__ == "__main__":
    best_scores = []
    best_solution = genetic_algorithm(init_solu, POPULATION_SIZE, best_scores)

    generation = [i for i in range(len(best_scores))]
    fitness = [s for s in best_scores]

    print(best_solution)
    distance, temps = calculate_D_and_T(best_solution, dist_matrix, dur_matrix, collection_time)
    print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")
    print(f"Fitness: {distance + temps}")
    print(has_duplicates(best_solution))
    print(f"Est-ce une bonne solution ? {is_permutation(best_solution, init_solu)}")

    distance, temps = calculate_D_and_T(init_solu, dist_matrix, dur_matrix, collection_time)
    print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")
    print(f"fitness sol initiale : {distance + temps}")

    plt.scatter(generation, fitness)
    plt.show()
