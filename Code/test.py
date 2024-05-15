import random
import pickle
import os
import time
import numpy as np
from matplotlib import pyplot as plt
import math
import itertools

ACCELERATED_MUTATION_THRESHOLD = 1000
POPULATION_SIZE = 500
WEIGHT_LIMIT = 5850
bad = 999999
ACCELERATED_MUTATION_NUMBER = 3
OLD_GENERATION = 200

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


os.chdir("../Probleme_Abers_2_bis")

with open("init_sol_Abers_pb2_bis.pickle", "rb") as f:
    init_solu_abers = pickle.load(f)

print(init_solu_abers)


def calculateDandT(l):
    distance = 0
    time = 0
    for i in range(len(l)):  # [0, 232]
        if i != len(l) - 1:
            distance += dist_matrix[l[i]][l[i + 1]]
            time += dur_matrix[l[i]][l[i + 1]]
        time += collection_time[l[i]]
    return distance, time

def fitness_iter(chemin) -> float:
    total_distance = 0
    total_weight = 0
    total_time = 0
    penalty = 0
    for i, j in itertools.islice(zip(chemin, chemin[1:]), len(chemin) - 1):
        total_distance += dist_matrix[i][j]
        total_time += dur_matrix[i][j]
        total_time += collection_time[i]
        total_weight += weight_list[i]
        total_weight = max(total_weight, 0)
        if total_weight > WEIGHT_LIMIT:
            penalty += bad
    total_time += collection_time[chemin[-1]]
    #if age > OLD_GENERATION:
    #    return (total_distance + total_time + penalty) * (1 + 0.01 * age)
    return total_distance + total_time + penalty

def fitness(chemin):
    total_distance = 0
    total_weight = 0
    total_time = 0
    penalty = 0
    for i in range(len(chemin) - 1):  # [0, 231]
        total_distance += dist_matrix[chemin[i]][chemin[i + 1]]

        total_time += dur_matrix[chemin[i]][chemin[i + 1]]
        total_time += collection_time[chemin[i]]

        total_weight += weight_list[chemin[i]]
        total_weight = max(total_weight, 0)
        if total_weight > WEIGHT_LIMIT:
            penalty += bad

    total_time += collection_time[chemin[-1]]
    return total_distance + total_time + penalty



notre_sol = [0, 80, 81, 82, 83, 84, 210, 85, 86, 87, 88, 78, 89, 90, 212, 219, 213, 214, 216, 109, 169, 172, 168, 16, 17, 18, 19, 40, 122, 133, 167, 34, 132, 131, 130, 35, 36, 37, 38, 39, 43, 44, 45, 46, 47, 48, 49, 166, 165, 164, 163, 103, 147, 4, 12, 68, 191, 146, 187, 158, 15, 127, 118, 159, 70, 20, 21, 200, 22, 31, 32, 33, 71, 173, 201, 174, 188, 73, 150, 190, 30, 181, 72, 66, 207, 77, 100, 101, 184, 115, 153, 160, 224, 113, 76, 156, 225, 209, 217, 215, 231, 114, 149, 148, 3, 208, 202, 58, 59, 60, 61, 185, 141, 140, 50, 9, 145, 51, 52, 53, 65, 55, 56, 139, 57, 138, 129, 137, 192, 120, 42, 175, 144, 143, 41, 13, 222, 14, 142, 227, 195, 62, 63, 54, 194, 196, 197, 6, 7, 8, 218, 198, 119, 193, 176, 64, 96, 5, 93, 121, 189, 128, 94, 23, 25, 26, 27, 28, 24, 126, 125, 124, 123, 95, 221, 229, 29, 1, 2, 199, 116, 67, 186, 10, 11, 117, 69, 104, 105, 106, 136, 135, 134, 107, 108, 220, 110, 92, 157, 111, 180, 179, 171, 228, 170, 161, 183, 182, 203, 154, 102, 205, 75, 97, 112, 152, 151, 226, 79, 162, 98, 99, 91, 206, 230, 74, 223, 232]
print(notre_sol)

distance, temps = calculateDandT(notre_sol)
print(f"Distance: {distance / 1000} km, Temps: {temps / 3600} h")
print(f"Fitness: {distance + temps}")
print(f"Fitness: {fitness(notre_sol)}")
print(f"Fitness_iter: {fitness_iter(notre_sol)}")

def check_list_equal(l1, l2):
    for i in range(len(l1)):
        if l1[i] != l2[i]:
            print(f"Change at index {i}: {l1[i]} != {l2[i]}")


teststssts = [0, 1, 2, 3, 232]
print(random.choice(teststssts[1:-1]))