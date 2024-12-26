import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import pickle
import argparse
from hypertab import HyperTabClassifier

parser = argparse.ArgumentParser(description = "Train Hypertab Model")
parser.add_argument("-t", "--task", type = str, default = "None", help = "posa, remosa")
parser.add_argument("-c", "--case", type = int, default = 1, help = "1, 2, 3, 4")
parser.add_argument("-p", "data_path", type = str, default = "None", help = "data_path")

args = parser.parse_args()

print(f"{args.task}, case: {args.case}")

with open(f'{args.data_path}/{args.task}/{args.case}/pickle/scaled_train.pkl', 'rb') as f:
    train_feature_data = pickle.load(f)
with open(f'{args.data_path}/{args.task}/{args.case}/pickle/train_label.pkl', 'rb') as f:
    train_labels = pickle.load(f)

# Test data load
with open(f'{args.data_path}/{args.task}/{args.case}/pickle/scaled_test.pkl', 'rb') as f:
    test_feature_data = pickle.load(f)
with open(f'{args.data_path}/{args.task}/{args.case}/pickle/test_label.pkl', 'rb') as f:
    test_labels = pickle.load(f)

# Ensure train_labels and test_labels are 1-dimensional
train_labels = np.ravel(train_labels)
test_labels = np.ravel(test_labels)

# Convert DataFrames to numpy arrays
train_feature_data = train_feature_data.to_numpy()
test_feature_data = test_feature_data.to_numpy()

DEVICE = "cuda:0"

print(f"{args.category}, scenario: {args.scenario}, downsampling: {args.downsampling}")

print("Hypertab Model")
# Train a Hypertab model (binary classification)
clf = HyperTabClassifier(0.5, device=DEVICE, test_nodes=100, epochs=100, hidden_dims=5)
clf.fit(train_feature_data, train_labels)

# Predict on the test set
predictions = clf.predict(test_feature_data)

# Calculate confusion matrix
cm = confusion_matrix(test_labels, predictions)

# Extract TP, FP, TN, FN from confusion matrix
TN, FP, FN, TP = cm.ravel()  # Ensure binary classification for ravel to work

# Print the values
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")

# Evaluate the model on the test set
test_accuracy = clf.score(test_feature_data, test_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print()

# Calculate and print F1 Score
f1 = f1_score(test_labels, predictions, average='weighted')
print(f"F1 Score: {f1*100:.2f}")

# Calculate and print AUROC
if len(np.unique(test_labels)) > 1:  # Ensure there are at least two classes in the test set
    probabilities = clf.predict_proba(test_feature_data)[:, 1]  # Probability for the positive class
    auroc = roc_auc_score(test_labels, probabilities)
    print(f"AUROC: {auroc*100:.2f}")
else:
    print("AUROC cannot be calculated due to a single class in test labels.")

print("=================")