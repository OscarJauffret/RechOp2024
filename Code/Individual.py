import pickle
import os

os.chdir("../Data/Probleme_Cholet_1_bis/")

with open("dist_matrix_Cholet_pb1_bis.pickle", "rb") as f:
    dist_matrix = pickle.load(f)

with open("dur_matrix_Cholet_pb1_bis.pickle", "rb") as f:
    dur_matrix = pickle.load(f)

with open("temps_collecte_Cholet_pb1_bis.pickle", "rb") as f:
    collection_time = pickle.load(f)

with open("weight_Cholet_pb1_bis.pickle", "rb") as f:
    weight_list = pickle.load(f)

WEIGHT_LIMIT = 5850
bad = 999999


class Individual:

    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.calculate_fitness()
        self.age = 0

    def increment_age(self) -> None:
        self.age += 1

    def get_age(self) -> int:
        return self.age

    def get_chromosome(self) -> list[int]:
        return self.chromosome

    def set_chromosome(self, chromosome: list[int]) -> None:
        self.chromosome = chromosome

    def get_fitness(self) -> float:
        return self.fitness

    def set_fitness(self, fitness) -> None:
        self.fitness = fitness

    def calculate_fitness(self) -> float:
        total_distance = 0
        total_weight = 0
        total_time = 0
        penalty = 0
        for i in range(len(self.chromosome) - 1):  # [0, 231]
            total_distance += dist_matrix[self.chromosome[i]][self.chromosome[i + 1]]

            total_time += dur_matrix[self.chromosome[i]][self.chromosome[i + 1]]
            total_time += collection_time[self.chromosome[i]]

            total_weight += weight_list[self.chromosome[i]]
            total_weight = max(total_weight, 0)
            if total_weight > WEIGHT_LIMIT:
                penalty += bad

        total_time += collection_time[self.chromosome[-1]]
        self.fitness = total_distance + total_time + penalty
        return self.fitness

