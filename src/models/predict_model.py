
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import joblib


def predict_model(model, data):
    preds = model.predict(data['test']['X'])
    accuracy = accuracy_score(data['test']['y'], preds)
    cm = confusion_matrix(data['test']['y'], preds)
    cr = classification_report(data['test']['y'], preds)
    return accuracy, cm, cr