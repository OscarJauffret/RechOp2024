import itertools
import multiprocessing
import random
import pickle
import os
import time
import math
import matplotlib.pyplot as plt
from utils import *

# Define the population size for the genetic algorithm
POPULATION_SIZE = 1000

# Define the weight limits for the two types of commodities
WEIGHT_LIMIT_TYPE1 = 10500
WEIGHT_LIMIT_TYPE2 = 7000

# Define the penalty value for exceeding the weight limits
BAD = 999999

# Define the mutation rates for normal and scramble mutations
MUTATION_RATE = 0.9
SCRAMBLE_MUTATION_RATE = 0.2

# Change directory to load dddata files, handle FileNotFoundError if the path is incorrect
try:
    os.chdir("../Data/Probleme_Abers_2/")
except FileNotFoundError:
    pass

# Load necessary data from pickle files
init_solu = load_data("init_sol_Abers_pb2.pickle")
dist_matrix = load_data("dist_matrix_Abers_pb2.pickle")
dur_matrix = load_data("dur_matrix_Abers_pb2.pickle")
collection_time = load_data("temps_collecte_Abers_pb2.pickle")
bilat_pairs = load_data("bilat_pairs_Abers_pb2.pickle")
weight_list = load_data("bonus_multi_commodity_Abers_pb2.pickle")

# Add the outlet node to the weight list with the maximum weight limits
weight_list[-1] = [-WEIGHT_LIMIT_TYPE1, -WEIGHT_LIMIT_TYPE2]
weight_list[-2] = [-WEIGHT_LIMIT_TYPE1, -WEIGHT_LIMIT_TYPE2]

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
    If the weight limits are exceeded, a penalty is added to the fitness.
    The function also finds the maximum distance between any two points in the solution and its index.

    Parameters:
    chemin (list): The solution to evaluate.

    Returns:
    tuple: The fitness and the index of the maximum distance.
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
    return ranked_solutions[:POPULATION_SIZE // 5]


def linear_base(variance, min_variance=100000, max_variance=250000, min_base=0.0005, max_base=1.01):
    """
    Calculate the base for the exponential decay function used in tournament selection.

    The base is calculated based on the variance of the fitness values in the population.

    Parameters:
    variance (float): The current variance of the population's fitness values.
    min_variance (float): The minimum threshold for variance below which exploration is encouraged.
    max_variance (float): The maximum threshold for variance above which exploitation is encouraged.
    min_base     (float): The base rate used when variance is at its minimum, promoting exploration.
    max_base (float): The base rate used when variance is at its maximum, promoting exploitation.

    Returns:
    float: The calculated base.
    """

    if variance <= min_variance:
        return min_base 
    elif variance >= max_variance:
        return max_base 
    else:
        return max_base + (min_base - max_base) * ((variance - min_variance) / (max_variance - min_variance))


def tournament_selection(population, variance, tournament_size=3):
    """
    Perform tournament selection.

    A number of individuals are randomly selected from the population and the best one is chosen.

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


def crossover(parent1: list[int], parent2: list[int], parent_1_worst_gene: int, parent_2_worst_gene: int) -> list[int]:
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
                child_set.add(bilat_pairs_dict.get(gene, 0)) 

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

    start = random.randint(1, len(individual) - 2 - length)
    end = start + length
    segment = individual[start:end]

    if random.random() < SCRAMBLE_MUTATION_RATE:
        subset = segment[1:-1] 
        random.shuffle(subset)  
        segment[1:-1] = subset 

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

    if random.random() < 0.2:
        if index == -1:
            index = random.randint(1, len(individual) - 2)
        if individual[index] in bilat_pairs_dict:
            individual[index] = bilat_pairs_dict[individual[index]] 

    return individual


def create_new_child(population, population_variance, stuck_generations, is_heavily_mutated_part):
    """
    Create a new child by selecting parents and applying genetic operations such as crossover and mutation.

    The mutation strategy varies depending on how long the population has been without improvements (stuck_generations) and
    whether the child is part of a heavily mutated portion of the population.

    Parameters:
    population (list): The current population of individuals.
    population_variance (float): The variance in fitness of the current population.
    stuck_generations (int): Number of generations without significant fitness improvements.
    is_heavily_mutated_part (bool): Flag indicating if heavier mutations should be applied.

    Returns:
    list: A new child created from selected parents with applied genetic operations.
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

def swap_mutation(individual):
    """
    Perform a swap mutation by exchanging two randomly selected genes in an individual.

    This function introduces genetic diversity by randomly selecting two genes within an individual and swapping their positions,
    which can lead to new traits in offspring.

    Parameters:
    individual (list): The individual on which to perform the swap mutation.

    Returns:
    list: The individual after the genes have been swapped.
    """
    
    first_index = random.randint(1, len(individual) - 2)
    second_index = random.randint(1, len(individual) - 2)
    individual[first_index], individual[second_index] = individual[second_index], individual[first_index]
    return individual

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
            
            population = selection(population, pool, stuck_generations)

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
            new_population.extend(individual[1] for individual in population[:population_size // 30])
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

    # Create lists for the generation numbers and fitness value
    generation = [i for i in range(len(best_scores))]
    fitness = [s for s in best_scores]

    # Print the best solution and its fitness value
    print(best_solution)

    #Calculate and print the distance and time for the best solution
    distance, temps = calculate_D_and_T(best_solution, dist_matrix, dur_matrix, collection_time)
    print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")

    # Print the fitness value of the best solution
    print(f"Fitness: {distance + temps}")

    # Check if the solution has duplicates and is a valid permutation
    print(f"Est-ce que la solution possède des doublons ? {has_duplicates(best_solution)}")
    print(f"Est-ce une bonne solution ? {is_permutation(best_solution, init_solu)}")

    # Calculate and print the distance and time for the initial solution
    distance, temps = calculate_D_and_T(init_solu, dist_matrix, dur_matrix, collection_time)
    print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")

    # Print the fitness value of the initial solution
    print(f"fitness sol initiale : {distance + temps}")

    # Save the results to a pickle file
    result = {
        "best_score": min(best_scores),
        "best_solution": best_solution,
        "distance_time": calculate_D_and_T(best_solution),
        "generation_best_scores": best_scores,
    }

    os.chdir("../../Code")
    with open("resultAbers_MultiComo.pkl", "wb") as f:
        pickle.dump(result, f)

    # Plot the fitness values over the generations
    plt.scatter(generation, fitness)
    plt.show()