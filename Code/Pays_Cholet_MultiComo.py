import itertools
import multiprocessing
import random
import pickle
import os
import time
from matplotlib import pyplot as plt
import math
from utils import *

# Define the population size for the genetic algorithm
POPULATION_SIZE = 1000

# Define the weight limits for the two types of commodities
WEIGHT_LIMIT_TYPE1 = 5850
WEIGHT_LIMIT_TYPE2 = 5850

# Define the penalty value for exceeding the weight limits
BAD = 999999 

# Define the mutation rates for normal and scramble mutations
MUTATION_RATE = 0.9
SCRAMBLE_MUTATION_RATE = 0.2

# Attempt to change the directory to where data files are stored, catching exceptions if the path doesn't exist
try:
    os.chdir("../Data/Probleme_Cholet_1_bis/")
except FileNotFoundError:
    pass

# Load necessary data from pickle files
init_solu = load_data("init_sol_Cholet_pb1_bis.pickle")
dist_matrix = load_data("dist_matrix_Cholet_pb1_bis.pickle")
dur_matrix = load_data("dur_matrix_Cholet_pb1_bis.pickle")
collection_time = load_data("temps_collecte_Cholet_pb1_bis.pickle")
weight_list = load_data("bonus_multi_commodity_Cholet_pb1_bis.pickle")

# Add the outlet node to the weight list with the maximum weight limits
weight_list[-1] = [-WEIGHT_LIMIT_TYPE1, -WEIGHT_LIMIT_TYPE2]
weight_list[-2] = [-WEIGHT_LIMIT_TYPE1, -WEIGHT_LIMIT_TYPE2]


def initialize_population(init_sol, population_size):
    """
    Initialize the population with the given initial solution and apply random mutations.

    Parameters:
    init_sol (list): The initial solution from which to derive the first population.
    population_size (int): The total number of individuals in the population.

    Returns:
    list: A list of individuals representing the initial population.
    """
    population = [init_sol.copy()]
    for _ in range(1, population_size):
        new_individual = mutation(init_sol.copy(), random.randint(1, 5))
        population.append(new_individual)
    return population


def fitness(chemin) -> tuple[float, int]:
    """
    Calculate the fitness of a solution.

    The fitness is calculated as the total distance of the solution.
    If the weight limits are exceeded, a penalty is added to the fitness. 
    The function also finds the maximum distance between any two points in the solution and its index.

    Parameters:
    chemin (list): The solution to evaluate.

    Returns:
    tuple: The fitness and the index of the maximum distance
    """
    total_distance = 0
    total_weight_type1 = 0
    total_weight_type2 = 0
    penalty = 0
    max_distance = 0
    index_max_distance = 0
    for i, j in itertools.islice(zip(chemin, chemin[1:]), len(chemin) - 1):
        current_distance = dist_matrix[i][j]
        if current_distance > max_distance:
            max_distance = current_distance
            index_max_distance = j
        total_distance += current_distance
        total_weight_type1 += weight_list[i][0]
        total_weight_type2 += weight_list[i][1]
        total_weight_type1 = max(0, total_weight_type1)
        total_weight_type2 = max(0, total_weight_type2)
        if total_weight_type1 > WEIGHT_LIMIT_TYPE1:
            penalty += BAD
        if total_weight_type2 > WEIGHT_LIMIT_TYPE2:
            penalty += BAD
    return total_distance + penalty, index_max_distance


def selection(population, pool):
    """
    Select the top individuals from the population based on fitness.

    Parameters:
    population (list): The current population of individuals.
    pool (multiprocessing.Pool): A pool of worker processes to parallelize fitness calculations.

    Returns:
    list: A subset of the population consisting of the top-performing individuals.
    """
    fitness_values = pool.map(fitness, population)
    ranked_solutions = sorted(zip(fitness_values, population), key=lambda x: x[0], reverse=False)
    return ranked_solutions[:POPULATION_SIZE // 6]


def linear_base(variance, min_variance=100000, max_variance=1000000, min_base=0.0005, max_base=1.01):
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
        return min_base 
    elif variance >= max_variance:
        return max_base
    else:
        return max_base + (min_base - max_base) * ((variance - min_variance) / (max_variance - min_variance))


def tournament_selection(population, variance, tournament_size=10):
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


def crossover(parent1: list[int], parent2: list[int], parent_1_worst_gene: int, parent_2_worst_gene:  int) -> list[int]:
    """
    Combine genes from two parents to create a new individual, focusing on cutting at the worst performing genes.

    Parameters:
    parent1, parent2 (list): The parent individuals.
    parent_1_worst_gene, parent_2_worst_gene (int): Indices of the worst performing genes in each parent.

    Returns:
    list: A new individual created by combining segments from both parents.
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

    for parent in [parent1, parent2]:
        for gene in parent:
            if gene not in child_set:
                child.append(gene)
                child_set.add(gene)
    return child


def mutation(individual, length=3):
    """
    Perform a mutation on an individual by segment rearrangement and potential scrambling within the segment.

    This function selects a segment of the individual based on the given length, optionally scrambles the contents of
    the segment, and then reinserts the (possibly scrambled) segment at a new position within the individual. This mutation
    is designed to explore new genetic configurations by altering the sequence of genes.

    Parameters:
    individual (list): The individual to mutate, represented as a list of gene indices.
    length (int): The length of the segment to extract and potentially scramble.

    Returns:
    list: The mutated individual with the segment rearranged.
    """

    start = random.randint(1, len(individual) - 2 - length)
    end = start + length
    segment = individual[start:end] 

    if random.random() < SCRAMBLE_MUTATION_RATE:
        subset = segment[1:-1]  
        random.shuffle(subset)  
        segment[1:-1] = subset 

    del individual[start:end]

    new_position = random.randint(1, len(individual) - 2)

    individual = individual[:new_position] + segment + individual[new_position:]
    return individual


def create_new_child(population, variance, stuck_generations, is_heavily_mutated_part):
    """
    Create a new child by combining segments from two parents and mutating the child.

    Parameters:
    population (list): The current population from which to select parents.
    variance (float): The variance of the population's fitness values.
    stuck_generations (int): The number of generations with no improvement.

    Returns:
    list: A new child individual created by crossover and mutation.
    """

    parent1 = tournament_selection(population, variance)
    parent2 = tournament_selection(population, variance)
    child = crossover(parent1[1], parent2[1], parent1[0][1], parent2[0][1])
    if not is_heavily_mutated_part:
        mutation_length = random.randint(2, 6) if stuck_generations > 10 else random.randint(1, 3)
    else:
        mutation_length = random.randint(6, 18) if stuck_generations > 10 else random.randint(3, 9)

    if random.random() < MUTATION_RATE:
        child = mutation(child, mutation_length)

    return child


def genetic_algorithm(init_sol, population_size, best_scores):
    """
    Perform the genetic algorithm

    The genetic algorithm creates a population of solutions and evolves it over a number of generations.
    The algorithm uses selection, crossover, and mutation to create new solutions.

    Parameters:
    init_sol (list): The initial solution.
    population_size (int): The size of the population.
    best_scores (list): A list to store the best scores of each generation.

    Returns:
    list: The best solution found by the genetic algorithm.
    """

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
            
            population_variance = calculate_variance(population) 
            print(f"Variance: {population_variance}")

            new_population = []
            new_population.extend(individual[1] for individual in population[:population_size // 50])
            
            while len(new_population) < population_size // 2: 
                child = create_new_child(population, population_variance, stuck_generations, False)
                new_population.append(child)

            while len(new_population) < population_size:
                child = create_new_child(population, population_variance, stuck_generations, True)
                new_population.append(child)

            population = new_population

    return min(population, key=fitness)

if __name__ == "__main__":
    # Initialize the best scores list to store the results
    best_scores = []

    # Run the genetic algorithm to optimize the solution
    best_solution = genetic_algorithm(init_solu, POPULATION_SIZE, best_scores)

    # Create lists for the generation numbers and fitness values
    generation = [i for i in range(len(best_scores))]
    fitness = [s[0] for s in best_scores]

    # Print the best solution and its fitness
    print(best_solution)

    # Calculate and print the distance and time for the best solution
    distance, temps = calculate_D_and_T(best_solution, dist_matrix, dur_matrix, collection_time)
    print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")

    # Print the fitness of the best solution
    print(f"Fitness: {distance + temps}")

    # Check if the best solution has duplicates and is a valid permutation
    print(f"Est-ce que la solution possède des doublons ? {has_duplicates(best_solution)}")
    print(f"Est-ce une bonne solution ? {is_permutation(best_solution, init_solu)}")

    # Calculate the distance and time for the initial solution
    distance, temps = calculate_D_and_T(init_solu, dist_matrix, dur_matrix, collection_time)
    print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")

    # Print the fitness of the initial solution
    print(f"fitness sol initiale : {distance + temps}")
    
    # Save the results to a pickle file
    result = {
        "best_score": min(best_scores),
        "best_solution": best_solution,
        "distance_time": calculate_D_and_T(best_solution),
        "generation_best_scores": best_scores,
    }

    os.chdir("../../Code")
    with open("resultCholet_MultiComo.pkl", "wb") as f:
        pickle.dump(result, f)

    # Plot the fitness values over the generations
    plt.scatter(generation, fitness)
    plt.show()