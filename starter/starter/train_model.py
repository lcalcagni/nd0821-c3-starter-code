'''Script to train machine learning model.
''' 

from sklearn.model_selection import train_test_split
import pandas as pd

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model

# Add code to load in the data.
data = pd.read_csv(r"../data/census_clean.csv", delimiter=';')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.
model = train_model(X_train, y_train)
pd.to_pickle(model, "../model/model.pkl")

# Save the encoder and the LabelBinarizer.
pd.to_pickle(encoder, "../model/encoder.pkl")
pd.to_pickle(lb, "../model/lb.pkl")