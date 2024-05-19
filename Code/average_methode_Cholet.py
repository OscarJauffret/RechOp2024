import subprocess
import concurrent.futures

def run_script():
    # Exécute le script et retourne la sortie
    result = subprocess.run(['python', '41800Cholet.py'], capture_output=True, text=True)
    score, path = result.stdout.strip().split(';')  # Changer la virgule par un point-virgule
    return (float(score), path.strip('[]').split(', '))  # Convertir le chemin en liste d'entiers si nécessaire

def main():
    results = []
    detailed_results = []  # Liste pour stocker les scores et les chemins pour la comparaison
    # Utilise ThreadPoolExecutor pour exécuter les scripts en parallèle
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_script) for _ in range(5)]
        for future in concurrent.futures.as_completed(futures):
            score, path = future.result()
            results.append(score)  # Stocker uniquement le score dans la liste des résultats            
            detailed_results.append((score, path))  # Stocker les scores et les chemins pour identifier le meilleur

    
    # Identifier la meilleure solution parmi les résultats détaillés
    if detailed_results:
        best_score, best_path = min(detailed_results, key=lambda x: x[0])
        print(f"Meilleur score: {best_score}, Meilleur chemin: {best_path}")
        print(f"Résultats individuels: {results}")
    else:
        print("Aucun résultat valide récupéré.")


if __name__ == "__main__":
    main()
