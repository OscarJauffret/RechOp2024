import subprocess
import concurrent.futures

def run_script():
    # Exécute le script et retourne la sortie
    result = subprocess.run(['python', 'test41800Massacre_para.py'], capture_output=True, text=True)
    return float(result.stdout.strip())

def main():
    results = []
    # Utilise ThreadPoolExecutor pour exécuter les scripts en parallèle
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_script) for _ in range(5)]
        for future in concurrent.futures.as_completed(futures):
            output = future.result()
            results.append(output)
    
    # Calcule la moyenne des résultats
    average = sum(results) / len(results)
    print(f"Moyenne des résultats: {average}")
    print(f"Résultats individuels: {results}")

if __name__ == "__main__":
    main()
