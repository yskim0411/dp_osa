import subprocess
import argparse

parser = argparse.ArgumentParser(description="Training non-DP model")
parser.add_argument("-t", "--task", type = str, default = "None", help = "posa, remosa")
parser.add_argument("-c", "--case", type = int, default = 1, help = "1, 2, 3, 4")

args = parser.parse_args()

data_path = ""

scripts = [
    "rf.py", 
    "logistic.py",
    "svm.py",
    "xgb.py",
    "cat.py",
    "hyper.py"
]

for script in scripts:
    command = ["python", script, "-t" , args.task, "-c", str(args.case), "-p", str(data_path)]
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Command {command} failed with return code {result.returncode}")
    else:
        print(f"Command {command} executed successfully")
        

subtab_train_script = "models/Subtab/train.py"
subtab_command = ["python", subtab_train_script, "-d", "osa"]
subtab_result = subprocess.run(subtab_command)
if subtab_result.returncode != 0:
    print(f"Subtab command {subtab_command} failed with return code {subtab_result.returncode}")
else:
    print(f"Subtab command {subtab_command} executed successfully")
