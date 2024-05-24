import itertools
import multiprocessing
import random
import pickle
import os
import time
import numpy as np
from matplotlib import pyplot as plt
import math
from utils import *

# The size of the population in each generation.
POPULATION_SIZE = 1000
# A placeholder value for bad solutions. This could be used to represent solutions that are not feasible or that have a very high cost.
bad = 999999
# The probability of mutation. This is the chance that an individual will undergo mutation in each generation.
MUTATION_RATE = 0.9

# Change the working directory to the data directory
try:
    os.chdir("../Data/Probleme_Abers_2_bis/")
except FileNotFoundError:
    pass
# Load the initial solution and transform it
init_solu = load_data("init_sol_Abers_pb2_bis.pickle")
init_solu = transform_data(init_solu)

# Load the other data files
dist_matrix = load_data("dist_matrix_Abers_pb2_bis.pickle")
weight_list = load_data("weight_Abers_pb2_bis.pickle")
dur_matrix = load_data("dur_matrix_Abers_pb2_bis.pickle")
collection_time = load_data("temps_collecte_Abers_pb2_bis.pickle")
bilat_pairs = load_data("bilat_pairs_Abers_pb2_bis.pickle")

# Create a dictionary from the bilateral pairs
bilat_pairs_dict = create_dict(bilat_pairs)    

def initialize_population(init_sol, population_size):
    """
    Initialize the population for the genetic algorithm.

    The initial population is created by performing random mutations on the initial solution.

    Parameters:
    init_sol (list): The initial solution.
    population_size (int): The size of the population to create.

    Returns:
    list: The initial population.
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

    return total_distance , index_max_distance


def selection(population, pool):
    """
    Select the best individuals from the population.

    The fitness of each individual is calculated and the individuals are sorted based on their fitness. 
    The top individuals are selected.

    Parameters:
    population (list): The population.
    pool (Pool): The multiprocessing pool.

    Returns:
    list: The selected individuals.
    """
    fitness_values = pool.map(fitness, population)
    ranked_solutions = sorted(zip(fitness_values, population), reverse=False)
    return ranked_solutions[:POPULATION_SIZE // 6]


def linear_base(variance, min_variance=50000, max_variance=180000, min_base=0.0005, max_base=1.01):
    """
    Calculate the base for the exponential decay function used in tournament selection.

    The base is calculated based on the variance of the fitness values in the population.

    Parameters:
    variance (float): The variance of the fitness values.
    min_variance (float): The minimum variance.
    max_variance (float): The maximum variance.
    min_base (float): The minimum base.
    max_base (float): The maximum base.

    Returns:
    float: The calculated base.
    """
    if variance <= min_variance:
        return min_base   
    elif variance >= max_variance:
        return max_base
    else:
        return max_base + (min_base - max_base) * ((variance - min_variance) / (max_variance - min_variance))


def tournament_selection(population, variance, tournament_size=10):
    """
    Perform tournament selection.

    A number of individuals are randomly selected from the population and the best one is chosen.

    Parameters:
    population (list): The population.
    variance (float): The variance of the fitness values.
    tournament_size (int): The number of individuals to select for the tournament.

    Returns:
    The selected individual.
    """
    n = len(population)
    base = linear_base(variance)
    weights = [math.exp(-base * i) for i in range(n)]
    return min(random.choices(population, weights=weights, k=tournament_size))


def crossover(parent1: list[int], parent2: list[int], parent_1_worst_gene: int, parent_2_worst_gene:  int) -> list[int]:
    """
    Perform crossover between two parents to create a child.

    The child is created by taking segments from each parent. The segments are chosen based on the worst genes of each parent. 
    The child inherits the rest of the genes from the parents in the order they appear.

    Parameters:
    parent1 (list[int]): The first parent.
    parent2 (list[int]): The second parent.
    parent_1_worst_gene (int): The index of the worst gene in the first parent.
    parent_2_worst_gene (int): The index of the worst gene in the second parent.

    Returns:
    list[int]: The child created by crossover.
    """
    # Determine the segments to take from each parent
    if parent_1_worst_gene < parent_2_worst_gene:
        segment1 = parent1[1:parent_1_worst_gene]
        segment2 = parent2[parent_1_worst_gene:parent_2_worst_gene]
        segment3 = parent1[parent_2_worst_gene:]
    else:
        segment1 = parent2[1:parent_2_worst_gene]
        segment2 = parent1[parent_2_worst_gene:parent_1_worst_gene]
        segment3 = parent2[parent_1_worst_gene:]

    # Initialize the child and a set to keep track of the genes in the child
    child = [0]
    child_set = set(child)

    # Add the genes from the segments to the child
    for segment in [segment1, segment2, segment3]:
        for gene in segment:
            if gene not in child_set:
                child.append(gene)
                child_set.add(gene)
                child_set.add(bilat_pairs_dict.get(gene, 0)) 

    # Add the remaining genes from the parents to the child
    for parent in [parent1, parent2]:
        for gene in parent:
            if gene not in child_set:
                child.append(gene)
                child_set.add(gene)
                child_set.add(bilat_pairs_dict.get(gene, 0)) 

    return child


def mutation(individual, length=3):
    """
    Perform mutation on an individual.

    A segment of the individual is selected and potentially shuffled. 
    The segment is then removed from the individual and inserted at a new random position. 
    Before and after the removal and insertion, the function also swaps the genes at the boundaries 
    of the segment with their pairs if they have any.

    Parameters:
    individual (list): The individual to mutate.
    length (int): The length of the segment to mutate.

    Returns:
    list: The mutated individual.
    """
    # Select a random segment of the individual
    start = random.randint(1, len(individual) - 2 - length)
    end = start + length
    segment = individual[start:end]

    # Possibly shuffle the segment
    if random.random() < 0.2:
        subset = segment[1:-1]
        random.shuffle(subset)
        segment[1:-1] = subset

    # Swap the genes at the boundaries of the segment with their pairs if they have any
    if start-1 != 0:
        individual = swap_with_pair(individual, start -1)
    individual = swap_with_pair(individual, start)
    individual = swap_with_pair(individual, end-1)
    individual = swap_with_pair(individual, end)

    # Remove the segment from the individual
    del individual[start:end]

    # Insert the segment at a new random position
    new_position = random.randint(1, len(individual) - 2)

    # Swap the genes at the boundaries of the new position with their pairs if they have any
    if new_position - 1 != 0:
        individual = swap_with_pair(individual, new_position - 1)
    individual = swap_with_pair(individual, new_position)

    # Insert the segment at the new position
    individual = individual[:new_position] + segment + individual[new_position:]

    return individual

def swap_with_pair(individual, index=-1):
    """
    Swap a gene in the individual with its pair.

    The function selects a gene at a given index or at a random index if no index is provided. 
    If the gene has a pair in the bilat_pairs_dict, it swaps the gene with its pair with a 20% probability.

    Parameters:
    individual (list): The individual to mutate.
    index (int, optional): The index of the gene to swap. If no index is provided, a random index is selected. Default is -1.

    Returns:
    list: The mutated individual.
    """
    # Select a random index if no index is provided
    if index == -1:
        index = random.randint(1, len(individual) - 2)

    # If the gene at the selected index has a pair, swap it with its pair with a 20% probability
    if individual[index] in bilat_pairs_dict:
        if random.random() < 0.2:
            individual[index] = bilat_pairs_dict[individual[index]]

    return individual



def genetic_algorithm(init_sol, population_size, best_scores):
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


if __name__ == "__main__":
    # Initialize lists to store the best scores and variances
    best_scores = []

    # Run the genetic algorithm and get the best solution
    best_solution = genetic_algorithm(init_solu, POPULATION_SIZE, best_scores)

    # Create lists for the generation numbers and fitness values
    generation = [i for i in range(len(best_scores))]
    fitness = [s for s in best_scores]

    # Print the best score ever obtained
    print(f"La meilleure solutions jamais obtenue est : {min(best_scores)}")

    # Print the best solution
    print(best_solution)

    # Calculate and print the distance and time for the best solution
    distance, temps = calculate_D_and_T(best_solution, dist_matrix, dur_matrix, collection_time)
    print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")

    # Print the fitness of the best solution
    print(f"Fitness: {distance + temps}")

    # Check and print if the best solution has duplicates
    print(has_duplicates(best_solution))

    # Check and print if the best solution is a permutation
    print(f"Est-ce une bonne solution ? {is_permutation(best_solution, init_solu)}")

    # Calculate and print the distance and time for the initial solution
    distance, temps = calculate_D_and_T(init_solu, dist_matrix, dur_matrix, collection_time)
    print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")

    # Print the fitness of the initial solution
    print(f"fitness sol initiale : {distance + temps}")

    # Plot the fitness values over generations
    plt.scatter(generation, fitness)
    plt.show()
