import subprocess
import os

# Define the range of input values you want to test
tournament_sizes = range(1, 12, 1)

# Define the path to your existing script
script_path = ".\Pays_Cholet_with_tournament_args.py"
python_path = "A:\CodePython\RechOp2024\Code\Scripts\python.exe"

def calculate_mean(lst):
    return sum(lst) / len(lst)

# Create a file to store the aggregated results
with open("aggregated_results_10_minutes_tournament_size.txt", "w") as results_file:
    # Run the script with each input value
    for tournament_size in tournament_sizes:
        # Use subprocess to run the script and capture the output*
        results = [None for _ in range(6)]
        for i in range(6):
            results[i] = subprocess.run([python_path, script_path, "--tournament_size", str(tournament_size)], capture_output=True, text=True)
            print("Tournament size: ", tournament_size, " - ", i, " - ", results[i].stdout)
            if results[i].returncode != 0:
                print(f"Subprocess for tournament size {tournament_size} failed with return code {results[i].returncode}")
                print(f"Standard error output: {results[i].stderr}")
                break

        if all(result.returncode == 0 for result in results):
            result = calculate_mean([float(result.stdout) for result in results])

            # Write the output to the results file
            results_file.write(f"Tournament size: {tournament_size}\n")
            results_file.write(str(result))
            results_file.write("\n")