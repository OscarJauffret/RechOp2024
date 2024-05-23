#def extract_best_scores(file_path):
#    best_scores = []  # List to store the best scores
#    with open(file_path, 'r') as file:
#        for line in file:
#            if "Best Score:" in line:
#                # Extract the score and convert it to integer
#                score = int(line.split(":")[1].strip())
#                best_scores.append(score)
#    return best_scores
#
## Example usage
#file_path = 'resultsAbers_Bis_MultiComo.txt'  # Replace 'your_file.txt' with your actual file path
#scores = extract_best_scores(file_path)
#min = min(scores)
#print(scores)
#print(f"best score = {min}")

def extract_best_scores_and_solutions(file_path):
    best_scores_solutions = {}  # Dictionnaire pour stocker les scores et les solutions associées
    current_score = None

    with open(file_path, 'r') as file:
        for line in file:
            if "Best Score:" in line:
                # Extraire le score et le convertir en entier
                current_score = int(line.split(":")[1].strip())
                best_scores_solutions[current_score] = None  # Initialiser la clé dans le dictionnaire
            elif "Best Solution:" in line and current_score is not None:
                # Extraire la solution, convertir en liste d'entiers
                solution = list(map(int, line.split(":")[1].strip().strip('[]').split(',')))
                best_scores_solutions[current_score] = solution

    # Trouver le score minimal (meilleur score)
    min_score = min(best_scores_solutions.keys())
    best_solution = best_scores_solutions[min_score]

    return min_score, best_solution

# Example usage
file_path = 'resultsAbers_Bis_MultiComo.txt'  # Remplacer 'your_file.txt' par le chemin réel de votre fichier
min_score, best_solution = extract_best_scores_and_solutions(file_path)
print(f"Best Score: {min_score}")
print(f"Best Solution: {best_solution}")
