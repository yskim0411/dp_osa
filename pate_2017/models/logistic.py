from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

def train(data, labels, ckpt_path, max_iter=100, random_state=42):
    """
    Train a Logistic Regression model and save the model checkpoint.

    :param data: Training data (features).
    :param labels: Corresponding labels.
    :param ckpt_path: Path to save the model checkpoint.
    :param max_iter: Maximum number of iterations for the solver.
    :param random_state: Random seed.
    :return: True if training and saving the model succeed.
    """
    # Initialize the Logistic Regression model
    model = LogisticRegression(max_iter=max_iter, random_state=random_state, solver='lbfgs')
    
    # Train the model
    model.fit(data, labels)
        
    # Save the trained model to the specified path
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    joblib.dump(model, ckpt_path)
    
    print(f"Model saved to {ckpt_path}")
    return True

def predict(data, ckpt_path):
    try:
        # Load the trained model from the checkpoint
        model = joblib.load(ckpt_path)
        
        # Make probability predictions
        probabilities = model.predict_proba(data)
        
        return probabilities
    except Exception as e:
        print(f"Failed to load model and predict: {e}")
        return None

def evaluate(data, labels, ckpt_path):
    try:
        # Load the trained model from the checkpoint
        model = joblib.load(ckpt_path)
        
        # Predict the class labels
        predicted_labels = model.predict(data)
        
        # Calculate accuracy
        accuracy = accuracy_score(labels, predicted_labels)
        
        # print(f"Predicted Labels: {predicted_labels}")
        # print(f"True Labels: {labels}")
        print(f"Model accuracy: {accuracy * 100:.2f}%")
        
        return accuracy
    except Exception as e:
        print(f"Failed to evaluate the model: {e}")
        return None
