import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from starter.ml.model import train_model, compute_model_metrics, inference
from starter.ml.data import process_data

df = pd.read_csv(os.path.join(os.getcwd(), r"starter/data/census_clean.csv"), delimiter=';')

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

train, test = train_test_split(df, test_size=0.3)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

def test_train_model():
    model = train_model(X_train, y_train)
    
    assert type(model) == RandomForestClassifier

def test_compute_model_metrics():
    model = train_model(X_train, y_train)
    y_pred = inference(model, X_train)
    
    metrics = compute_model_metrics(y_train, y_pred)
    assert len(metrics)==3
    assert type(metrics) == tuple
    for metric in metrics:
        assert metric >=0 and metric <= 1

def test_inference():
    model = train_model(X_train, y_train)
    y_pred = inference(model, X_train)
    
    assert np.all((y_pred==0)|(y_pred == 1)) == True

if __name__ == "__main__":
    test_train_model()
    test_compute_model_metrics()
    test_inference()
