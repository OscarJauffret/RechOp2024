from numba import cuda
import numpy as np
import time
import itertools

# Exemple de matrice de distance pour la démonstration
dist_matrix = np.random.rand(233, 233).astype(np.float32)

@cuda.jit
def fitness_kernel(individuals, dist_matrix, fitnesses):
    """
    CUDA kernel to calculate the fitness of multiple individuals.
    
    Parameters:
    individuals (ndarray): The solutions to evaluate.
    dist_matrix (ndarray): The distance matrix.
    fitnesses (ndarray): Array to store the fitness results.
    """
    idx = cuda.grid(1)
    n_individuals, n_cities = individuals.shape
    
    if idx < n_individuals:
        total_distance = 0.0
        for i in range(n_cities - 1):
            from_city = individuals[idx, i]
            to_city = individuals[idx, i + 1]
            total_distance += dist_matrix[from_city, to_city]
        
        fitnesses[idx] = total_distance

def fitness_CPU(chemin) -> tuple[float, int]:
    total_distance = 0
    for i, j in itertools.islice(zip(chemin, chemin[1:]), len(chemin) - 1):
        total_distance += dist_matrix[i][j]

    return total_distance

def fitness_GPU(population):
    n_individuals, n_cities = population.shape
    
    # Convert the population to a Numpy array
    population_np = np.array(population, dtype=np.int32)
    
    # Allocate memory on the device
    population_device = cuda.to_device(population_np)
    dist_matrix_device = cuda.to_device(dist_matrix)
    fitnesses_device = cuda.device_array(n_individuals, dtype=np.float32)  # Store the fitness results

    # Configure blocks and threads
    threads_per_block = 128
    blocks_per_grid = (n_individuals + threads_per_block - 1) // threads_per_block
    #print("Blocks per grid:", blocks_per_grid)
    # Launch kernel
    #start = time.time()
    fitness_kernel[blocks_per_grid, threads_per_block](population_device, dist_matrix_device, fitnesses_device)
    #print("Time to compute fitness on GPU:", time.time() - start)
    # Copy the results back to host
    fitnesses_result = fitnesses_device.copy_to_host()

    # Return the fitness values
    return fitnesses_result

# Exemple d'utilisation
def create_population(population_size, individual_size):
    return np.random.randint(0, 233, size=(population_size, individual_size))

# Créer une population de 1000 individus
population = create_population(10000, 233)


# Exemple d'utilisation
print("CPU")
list = []
for _ in range(1):
    popu = create_population(10000, 233)
    start=time.time()
    for individual in popu:
        fitness_value = fitness_CPU(individual)
    list.append(time.time()-start)
    #print(f"Tme in boucle {time.time()-start}")

average = sum(list)/len(list)
total_time = sum(list)
print(f"Average time CPU: {average}")
print(f"Total time CPU: {total_time}")

print("GPU")
list = []
for _ in range(1):
    popu = create_population(10000, 233)
    start=time.time()
    fitnesses = fitness_GPU(population)
    list.append(time.time()-start)
    #print(f"Tme in boucle {time.time()-start}")

average = sum(list)/len(list)
total_time = sum(list)
print(f"Average time GPU: {average}")
print(f"Total time GPU: {total_time}")
