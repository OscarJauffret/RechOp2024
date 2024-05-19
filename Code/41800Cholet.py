import itertools
import multiprocessing
import random
import pickle
import os
import time
from matplotlib import pyplot as plt
import math

# Define global constants for the genetic algorithm parameters and environment settings
POPULATION_SIZE = 1000
WEIGHT_LIMIT = 5850
BAD = 999999  # Penalty value for exceeding weight limits
MUTATION_RATE = 0.9

# Attempt to change the directory to where data files are stored, catching exceptions if the path doesn't exist
try:
    os.chdir("../Data/Probleme_Cholet_1_bis/")
except FileNotFoundError:
    pass

# Load initial solutions and related data structures from serialized pickle files
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
    """
    Initialize the population with the given initial solution and apply random mutations.

    Parameters:
    init_sol (list): The initial solution from which to derive the first population.
    population_size (int): The total number of individuals in the population.

    Returns:
    list: A list of individuals representing the initial population.
    """
    print(f"Génération 0: Solution initiale: {fitness(init_sol)} {init_sol}")
    population = [init_sol.copy()]
    for _ in range(1, population_size):
        new_individual = mutation(init_sol.copy(), random.randint(1, 5))
        population.append(new_individual)
    return population

def fitness(chemin):
    """
    Calculate the fitness of a solution, combining total distance and penalties for excess weight.

    Parameters:
    chemin (list): A route or path represented as a list of node indices.

    Returns:
    tuple: A tuple containing the total distance (float) and the index of the furthest node (int).
    """
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
        if total_weight > WEIGHT_LIMIT:
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

def calculateVariance(population):
    """
    Calculate the variance of fitness values within the population, used to adjust mutation parameters dynamically.

    Parameters:
    population (list): The current population whose fitness variance is to be calculated.

    Returns:
    float: The variance of the fitness values within the population.
    """
    mean = sum(sol[0][0] for sol in population) / len(population)
    return sum((sol[0][0] - mean) ** 2 for sol in population) / len(population)

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
        return min_base     # small base to promote exploration 
    elif variance >= max_variance:
        return max_base     # Tall base to promote exploitation. (Give higher weight to best solutions)
    else:
        return max_base + (min_base - max_base) * ((variance - min_variance) / (max_variance - min_variance))

# Tournament selection based on calculated weights
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

# Cut On Worst L+R Crossover
def crossover(parent1: list[int], parent2: list[int], parent_1_worst_gene: int, parent_2_worst_gene:  int) -> list[int]:
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

    # Add remaining elements from parents to complete the child
    for parent in [parent1, parent2]:
        for gene in parent:
            if gene not in child_set:
                child.append(gene)
                child_set.add(gene)
    return child

def mutation(individual, length=3):
    """
    Perform segment mutation on an individual by randomly relocating a segment.

    Parameters:
    individual (list): The individual to mutate.
    length (int): The length of the segment to mutate.

    Returns:
    list: The mutated individual with a segment randomly relocated.
    """
    # Select the starting position of the segment to mutate, ensuring it does not affect the first or last gene
    start = random.randint(1, len(individual) - 2 - length)
    end = start + length
    segment = individual[start:end]  # Extract the segment to be relocated
    del individual[start:end]  # Remove the segment from its original location
    new_position = random.randint(1, len(individual) - 2)  # Select a new position for the segment
    individual = individual[:new_position] + segment + individual[new_position:]  # Reinsert the segment
    return individual

def swap_mutation(individual):
    """
    Introduce variability by swapping two randomly selected genes within an individual.

    Parameters:
    individual (list): The individual to mutate.

    Returns:
    list: The mutated individual with two genes swapped.
    """
    # Select two random indices for swapping
    i, j = random.sample(range(1, len(individual) - 1), 2)
    # Swap the genes at the selected indices
    individual[i], individual[j] = individual[j], individual[i]
    return individual

def genetic_algorithm(init_sol, population_size, best_scores):
    """
    Optimize the solution using a genetic algorithm framework with selection, crossover, and mutation operations.

    Parameters:
    init_sol (list): The initial solution or seed for the population.
    population_size (int): The size of the population to maintain.
    best_scores (list): A list to record the best scores achieved during the optimization.

    Returns:
    list: The best solution found during the genetic algorithm execution.
    """

    population = initialize_population(init_sol, population_size)   # Initialize the population
    start_time = time.time()
    generation = 0
    stuck_generations = 0
    previous_score = 0

    with multiprocessing.Pool() as pool:    # Use multiprocessing to handle parallel computation of fitness evaluations
        while time.time() - start_time < 600:   # Run until 10 minutes
            
            population = selection(population, pool)
            best_score = population[0][0][0]
            best_scores.append((best_score,population[0][1]))

             # Check if the best score has changed from the previous generation
            if best_score == previous_score:
                stuck_generations += 1
            else:
                stuck_generations = 0
                previous_score = best_score
            generation += 1
            print(f"Generation {generation}: {best_score}", end=" ")
            
            # Calculate the variance of fitness scores in the population for dynamic adaptations
            population_variance = calculateVariance(population) 
            print(f"Variance: {population_variance}")
            variances.append(population_variance)
            
            # Create a new population starting with the best performing individuals
            new_population = []
            new_population.extend(individual[1] for individual in population[:population_size // 50])   #1000/50 = 20 best individuals
            
            # Fill a part of the new population with offspring from selected parents
            while len(new_population) < population_size // 2:   #1000/2 = 500 child form best parents choose with tournament selection
                # Perform tournament selection based on current population variance
                parent1, parent2 = tournament_selection(population, population_variance), tournament_selection(population, population_variance)
                child = crossover(parent1[1], parent2[1], parent1[0][1], parent2[0][1]) # Create a child through crossover
                
                # Determine mutation strength based on the number of stagnant generations
                mutation_length = random.randint(2, 6) if stuck_generations > 10 else random.randint(1, 3)
                if random.random() < MUTATION_RATE:
                    child = mutation(child, mutation_length)    # Mutate the child
                new_population.append(child)    # Add the new child to the population

            # Continue populating until the desired population size is reached
            while len(new_population) < population_size:
                # Perform tournament selection based on current population variance
                parent1, parent2 = tournament_selection(population, population_variance), tournament_selection(population, population_variance)
                child = crossover(parent1[1], parent2[1], parent1[0][1], parent2[0][1]) # Create a child through crossover
                
                # Determine mutation strength based on the number of stagnant generations
                mutation_length = random.randint(6, 18) if stuck_generations > 10 else random.randint(3, 9)
                if random.random() < MUTATION_RATE:
                    child = mutation(child, mutation_length)    # Mutate the child
                new_population.append(child)    # Add the new child to the population

            population = new_population     # Replace the old population with the new one

    return min(population, key=fitness)

def calculateDandT(l):
    """
    Calculate the total distance and time for a given route.

    Parameters:
    l (list): A route represented as a list of node indices.

    Returns:
    tuple: Total distance and time for the route.
    """

    distance = 0
    time = 0
    for i in range(len(l)):
        if i != len(l) - 1:
            distance += dist_matrix[l[i]][l[i + 1]]
            time += dur_matrix[l[i]][l[i + 1]]
        time += collection_time[l[i]]
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

    return sorted(l) == list(range(233)) and l[0] == 0 and l[-1] == 232


if __name__ == "__main__":
    best_scores = []
    variances = []
    best_solution = genetic_algorithm(init_solu, POPULATION_SIZE, best_scores)

    generation = [i for i in range(len(best_scores))]
    fitness = [s[0] for s in best_scores]

    print(f"La meilleure solutions jamais obtenue est : {min(best_scores)[0]} avec le chemin : {min(best_scores)[1]}")
    #print(f"{min(best_scores)[0]};{min(best_scores)[1]}")
    print(best_solution)
    distance, temps = calculateDandT(best_solution)
    print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")
    print(f"Fitness: {distance + temps}")
    print(has_duplicates(best_solution))
    print(f"Est-ce une bonne solution ? {ispermutation(best_solution)}")
    distance, temps = calculateDandT(init_solu)
    print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")
    print(f"fitness sol initiale : {distance + temps}")

    # Plot a graph to see the evolution 
    plt.scatter(generation, fitness)
    plt.show()
