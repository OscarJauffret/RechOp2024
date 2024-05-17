# script_maitre.py
import subprocess
import os
import pickle
import time

python_path = "A:\CodePython\RechOp2024\Code\Scripts\python.exe"

def run_script_and_save_results(script_path, output_file, max_runs=10):
    run_count = 0
    while run_count < max_runs:
        # Exécute le script cible
        print(os.getcwd())
        subprocess.run([python_path, script_path])

        # Charger le résultat du fichier de résultat temporaire
        with open("result.pkl", "rb") as f:
            result = pickle.load(f)

        # Enregistrer les résultats dans le fichier de sortie
        with open(output_file, "a") as f:
            f.write(f"Run {run_count + 1}:\n")
            f.write(f"Best Score: {result['best_score']}\n")
            f.write(f"Best Solution: {result['best_solution']}\n")
            f.write(f"Distance and Time: {result['distance_time']}\n")
            f.write(f"Generation Best Scores: {result['generation_best_scores']}\n")
            f.write("\n")

        run_count += 1
        print(f"Run {run_count} completed.")

        # Optionnel : Attendre un moment avant de relancer le script
        time.sleep(1)

if __name__ == "__main__":
    script_path = "Pays_Abers.py"  # Chemin vers le script de l'algorithme
    output_file = "resultsAbers.txt"  # Fichier où les résultats seront enregistrés
    max_runs = 10  # Nombre maximum d'exécutions

    # Supprimer le fichier de résultats précédent s'il existe
    if os.path.exists(output_file):
        os.remove(output_file)

    # Lancer le script et enregistrer les résultats
    run_script_and_save_results(script_path, output_file, max_runs)
