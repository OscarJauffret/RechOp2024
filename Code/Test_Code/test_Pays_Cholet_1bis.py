import pickle
import random
import numpy as np
import matplotlib.pyplot as plt



# Chemins vers les fichiers pickle
files = {
    #'bonus_multi_commodity': './Input_Data/Probleme_Cholet_1_bis/bonus_multi_commodity_Cholet_pb1_bis.pickle',
    'dist_matrix': '../Data/Probleme_Cholet_1_bis/dist_matrix_Cholet_pb1_bis.pickle',
    'dur_matrix': '../Data/Probleme_Cholet_1_bis/dur_matrix_Cholet_pb1_bis.pickle',
    'init_sol': '../Data/Probleme_Cholet_1_bis/init_sol_Cholet_pb1_bis.pickle',
    'temps_collecte': '../Data/Probleme_Cholet_1_bis/temps_collecte_Cholet_pb1_bis.pickle',
    'weight': '../Data/Probleme_Cholet_1_bis/weight_Cholet_pb1_bis.pickle'
}

# Fonction pour charger un fichier pickle
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# # Chargement des données
# data = {key: load_pickle(path) for key, path in files.items()}

# # Affichage de la structure des données chargées pour voir si tout est ok
# for key, value in data.items():
#     print(f"{key}: Type={type(value)}, Length={len(value)}")
#     if isinstance(value, list) or isinstance(value, dict):
#         print(f"Sample data for {key}: {value[:5]} \n")  # Affiche les premiers éléments pour les listes ou dictionnaires
#     else:
#         print(f"Content for {key}: {value} \n\n\n")


#Chargement des données
data = {key: load_pickle(path) for key, path in files.items()}
dist_matrix = data['dist_matrix']
dur_matrix = data['dur_matrix']
weights = data['weight']
collect_times = data['temps_collecte']
#init_sol = data['init_sol']

#Test pour voir si tout est ok
#print(f"dist : {dist_matrix}\n\n dur : {dur_matrix}\n\n weights : {weights}\n\n collect : {collect_times}\n\n init_sol : {init_sol}")

#Parametre
bad = -99999
num_solutions = 2
num_generation = 100
#n_nodes = len(weights)   
capacity = 5850
init_sol = [1,2,3,4,5]      #Solut de base simple pour faire des tests
n_nodes = 4     #Pareil

#Générer les solutions de départ 
population = [list(init_sol) for _  in range(num_solutions)]
print('\n\n Population = ', population)
for sol in population:
    random.shuffle(sol)     #On mélange les solution pour la diversité 
#Ou alors faut utiliser les org_tool pour générer plusieurs solutions de départ ? 
print('\n\n Population apres shuffle = ', population)

def fitness(chemin):
    total_dsitance = 0
    total_weight = 0
    total_time = 0
    penalty = 0
    for i in range(1, len(chemin)):
        total_dsitance += dist_matrix[chemin[i-1]][chemin[i]]
        dist_matrix[chemin[i-1]][chemin[i]]
        total_weight += weights[chemin[i]]
        total_time += collect_times[chemin[i]]
        if total_weight > capacity:
            penalty += bad
            #ou mettre une penalyté variable genre 
            #penalty += (total_weight - capacity) * 10
    return total_dsitance + total_time + penalty

#print(dist_matrix[1][1], ' ', dist_matrix[1][2], ' ', dist_matrix[1][3] )

#Fonction de crossover en un point
def crossover(parent1, parent2):
    print(f"Parent1 = {parent1} et Parent2 = {parent2}")
    cut = random.randint(1, n_nodes -1)     #Coupe aléatoirement la solution en 2
    child1 = parent1[:cut] + [x for x in parent2 if x not in parent1[:cut]]
    child2 = parent2[:cut] + [x for x in parent1 if x not in parent2[:cut]]
    print(f"Child1 = {child1} et Child2 = {child2}")
    return child1, child2
crossover(population[0], population[1])
#Pas top car pour des solutions for semblable, ça change rien 
#Interet de passer sur une autre méthode de crossover 

def pmx(parent1, parent2):
    size = len(parent1)
    p1, p2 = [-1] * size, [-1] * size
    child1, child2 = [-1] * size, [-1] * size

    # Initialize position maps
    for i in range(size):
        p1[parent1[i]] = i
        p2[parent2[i]] = i

    # Choose crossover points
    cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))
    print(f"Crossover points: {cxpoint1} to {cxpoint2}")

    # Crossover between cxpoint1 and cxpoint2
    for i in range(cxpoint1, cxpoint2 + 1):
        child1[i], child2[i] = parent2[i], parent1[i]
        p1[parent2[i]], p2[parent1[i]] = i, i

    # Complete the children
    for i in range(size):
        if not cxpoint1 <= i <= cxpoint2:
            # Fix child1
            n = parent1[i]
            while child1[p1[n]] != -1:
                n = parent1[p1[n]]
            child1[i] = n

            # Fix child2
            n = parent2[i]
            while child2[p2[n]] != -1:
                n = parent2[p2[n]]
            child2[i] = n

    return child1, child2


# Test the PMX crossover
#parent1, parent2 = deepcopy(population[0]), deepcopy(population[1])
#child1, child2 = pmx(parent1, parent2)  # Use slicing to avoid modifying the original parents
#print(f"Child1 = {child1}, Child2 = {child2}")

def mutate(chemin):
    i, j = random.sample(range(n_nodes), 2)     #On prend 2 points aléatoire dans la solution
    chemin[i], chemin[j] = chemin[j], chemin[i]     #On echange les 2 valeurs 
    #Attention vue notre grand nombre de noeud, il faudra faire plus que 1 éhcnage dans une solution
    return chemin

print(f"init_sol = {init_sol}\n")
mutate(init_sol)
print(f"init_sol apres mutation = {init_sol}")



