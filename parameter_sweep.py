import subprocess
from tqdm import tqdm

# Read sim_pars.txt file
with open('sim_pars.txt', 'r') as file:
    lines = file.readlines()

# Loop through each line and run main.py with corresponding arguments
for line in tqdm(lines):
    # Split the line into individual parameters
    parameters = line.strip().split()

    # The first element is the script name (main.py)
    script_name = "main.py"

    # Create the command to run main.py with the parameters
    command = ["python", script_name] + parameters
    print(command)

    # Run the command
    subprocess.run(command)