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

The final evaluation of the model achieved these results:
- precision: 0.8264833127317676
- recall: 0.6821833949751307
- fbeta: 0.7474324041081534

On the other hand, the performance of the model on slices of the data is located in file `./starter/starter/slice_performance.txt`.

## Ethical Considerations

It is woth mentioning that the data used in this model is from 1996, which means that the predictions made by this model are based on data taken more than 25 years ago.

## Caveats and Recommendations

In order to make this model more robust, it is suggested to implemend cross-validation in the future.

