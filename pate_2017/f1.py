import re

file_path = "remosa50.txt"

# Reading the file content
with open(file_path, 'r') as file:
    content = file.read()

# Extracting all F1 scores using regex
f1_scores = re.findall(r"Model F1 Score: ([\d\.]+)", content)

# Converting the F1 scores to floats
f1_scores = [float(score) for score in f1_scores]

# Calculating the average F1 score
if f1_scores:
    average_f1_score = sum(f1_scores) / len(f1_scores)
    print(f"f1: {average_f1_score}")