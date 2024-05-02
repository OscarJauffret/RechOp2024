import subprocess
import os

# Define the range of input values you want to test
mutation_rates = range(1, 7, 1)

# Define the path to your existing script
script_path = ".\Pays_Cholet_with_mutation_args.py"
python_path = "A:\CodePython\RechOp2024\Code\Scripts\python.exe"

def calculate_mean(lst):
    return sum(lst) / len(lst)

# Create a file to store the aggregated results
with open("aggregated_results_10_minutes.txt", "w") as results_file:
    # Run the script with each input value
    for mutation_rate in mutation_rates:
        # Use subprocess to run the script and capture the output*
        results = [None for _ in range(6)]
        for i in range(6):
            results[i] = subprocess.run([python_path, script_path, "--mutation_rate", str(mutation_rate)], capture_output=True, text=True)
            print("Mutation rate: ", mutation_rate, " - ", i, " - ", results[i].stdout)
            if results[i].returncode != 0:
                print(f"Subprocess for mutation_rate {mutation_rate} failed with return code {results[i].returncode}")
                print(f"Standard error output: {results[i].stderr}")
                break

        if all(result.returncode == 0 for result in results):
            result = calculate_mean([float(result.stdout) for result in results])

            # Write the output to the results file
            results_file.write(f"Mutation rate: {mutation_rate}\n")
            results_file.write(str(result))
            results_file.write("\n")