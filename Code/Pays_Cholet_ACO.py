import numpy as np
import pickle
import os

os.chdir("../Data/Probleme_Cholet_1_bis/")

# Load necessary data
with open("init_sol_Cholet_pb1_bis.pickle", "rb") as f:
    init_solu = pickle.load(f)

np.savetxt('init_solu.txt', np.array(init_solu), fmt='%d')

with open("dist_matrix_Cholet_pb1_bis.pickle", "rb") as f:
    dist_matrix = pickle.load(f)
np.savetxt('dist_matrix.txt', np.array(dist_matrix), fmt='%f')

with open("dur_matrix_Cholet_pb1_bis.pickle", "rb") as f:
    dur_matrix = pickle.load(f)
np.savetxt('dur_matrix.txt', np.array(dur_matrix), fmt='%f')

with open("temps_collecte_Cholet_pb1_bis.pickle", "rb") as f:
    collection_time = pickle.load(f)
np.savetxt('collection_time.txt', np.array(collection_time), fmt='%f')

with open("weight_Cholet_pb1_bis.pickle", "rb") as f:
    weight_list = pickle.load(f)


def initialize_population(init_sol, population_size):
    population = np.tile(init_sol, (population_size, 1))
    for individual in population:
        np.random.shuffle(individual[1:-1])
    return population


def distance(node1, node2):
    return dist_matrix[node1][node2]


def ant_colony_optimization(nodes, n_ants, n_iterations, alpha, beta, evaporation_rate, Q):
    n_points = len(nodes)
    pheromone = np.ones((n_points, n_points))
    best_path = None
    best_path_length = np.inf

    for iteration in range(n_iterations):
        print(f"Génération {iteration}")
        paths = []
        path_lengths = []

        for ant in range(n_ants):
            print(f"Fourmi {ant}")
            visited = [False] * n_points
            current_point = np.random.randint(n_points)
            visited[current_point] = True
            path = [current_point]
            path_length = 0

            while False in visited:
                unvisited = np.where(np.logical_not(visited))[0]        # Liste des points non visités
                probabilities = np.zeros(len(unvisited))

                # pheromone[current_point, unvisited_point] est la même chose que pheromone[current_point][unvisited_point]
                for i, unvisited_point in enumerate(unvisited):
                    epsilon = 1e-10  # small constant to prevent division by zero
                    probabilities[i] = pheromone[current_point][unvisited_point] ** alpha / (
                                distance(nodes[current_point], nodes[unvisited_point]) ** beta + epsilon)

                probabilities /= np.sum(probabilities)

                next_point = np.random.choice(unvisited, p=probabilities)
                path.append(next_point)
                path_length += distance(nodes[current_point], nodes[next_point])
                visited[next_point] = True
                current_point = next_point

            paths.append(path)
            path_lengths.append(path_length)

            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length

        pheromone *= evaporation_rate

        for path, path_length in zip(paths, path_lengths):
            for i in range(n_points - 1):
                pheromone[path[i], path[i + 1]] += Q / path_length
            pheromone[path[-1], path[0]] += Q / path_length


# Example usage:
points = init_solu
#points =np.random.rand(10, 3)
ant_colony_optimization(points, n_ants=500, n_iterations=1000, alpha=1, beta=1, evaporation_rate=0.5, Q=1)