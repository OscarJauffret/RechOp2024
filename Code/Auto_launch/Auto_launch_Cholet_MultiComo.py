import subprocess
import os
import pickle
import time

python_path = "python"

def run_script_and_save_results(script_path, output_file, max_runs=100):
    run_count = 0
    while True:
        # Exécute le script cible
        subprocess.run([python_path, script_path])

        # Charger le résultat du fichier de résultat temporaire
        with open("resultCholet_MultiComo.pkl", "rb") as f:
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
    script_path = "Pays_Cholet_MultiComo.py"  # Chemin vers le script de l'algorithme
    output_file = "resultCholet_MultiComo.txt"  # Fichier où les résultats seront enregistrés
    # max_runs = 10  # Nombre maximum d'exécutions

    # Supprimer le fichier de résultats précédent s'il existe
    # if os.path.exists(output_file):
    #     os.remove(output_file)

    # Lancer le script et enregistrer les résultats
    run_script_and_save_results(script_path, output_file)
