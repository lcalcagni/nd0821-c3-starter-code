# Model Card

## Model Details
This random forest classifier model has been developed by Laura Calcagni. It uses the  `RandomForestClassifier` from `sklearn` library.

## Intended Use
The goal of this model is to be able to predict if someone annual income exceeds $50K given census data.

## Training Data
The data used in this project is the Census Income Data Set provided by Udacity and located in file `./starter/data/census_clean.csv`.
The training subset is the 80% of the data.
## Evaluation Data
20% of the census-clean.csv data.

## Metrics
The performance of the model was evaluated using the following metrics:
- precision
- recall
- fbeta

The performance of the model on slices of the data is located in file `./starter/starter/slice_performance.txt`.
