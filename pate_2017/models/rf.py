from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import joblib
import os
import numpy as np

def train(data, labels, ckpt_path, n_estimators=100, max_depth=None, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, )
    model.fit(data, labels)
    
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    joblib.dump(model, ckpt_path)
    
    print(f"Model saved to {ckpt_path}")
    return True

def predict(data, ckpt_path):
    try:
        model = joblib.load(ckpt_path)
        probabilities = model.predict_proba(data)
        return probabilities
    except Exception as e:
        print(f"Failed to load model and predict: {e}")
        return None

def evaluate(data, labels, ckpt_path):
    try:
        model = joblib.load(ckpt_path)
        print(ckpt_path)
        predicted_labels = model.predict(data)
        probabilities = model.predict_proba(data)
        
        accuracy = accuracy_score(labels, predicted_labels)
        
        # ROC AUC 계산 조건 추가
        if len(np.unique(labels)) > 1:
            # 두 개 이상의 클래스가 있을 때만 AUROC 계산
            auroc = roc_auc_score(labels, probabilities[:, 1])
            print(f"Model AUROC: {100*auroc:.2f}")
            
            f1 = f1_score(labels, predicted_labels, average='weighted')  # Weighted 평균 F1 Score
            print(f"Model F1 Score: {100*f1:.2f}")
        else:
            # auroc = None
            auroc = 0
            f1 = 0
            print("AUROC is not defined for a single class in y_true.")

        print(f"Model accuracy: {accuracy * 100:.2f}%")
        return accuracy, auroc, f1
    except Exception as e:
        print(f"Failed to evaluate the model: {e}")
        return None