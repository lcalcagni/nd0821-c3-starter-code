import pandas as pd

from ml.model import inference, compute_model_metrics

from ml.data import process_data

df = pd.read_csv("../data/census_clean.csv", delimiter=';')

model = pd.read_pickle("../model/model.pkl")
encoder = pd.read_pickle("../model/encoder.pkl")
lb = pd.read_pickle("../model/lb.pkl")

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


def slice_performance(model, df, cat_features, encoder, lb):

    file = open('slice_performance.txt','w')

    for cat in cat_features:
        file.write(f'Category: {cat}')
        file.write('\n')
        file.write('\n')
    
        for value in df[cat].unique():
            file.write(value)
            file.write('\n')

            df_cat_col = df[df[cat]==value]

            X_test, y_test, encoder, lb = process_data(
                df_cat_col, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
            )

            y_pred = inference(model, X_test)

            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

            file.write('Precision   {}\n'.format(precision))
            file.write('Recall  {}\n'.format(recall))
            file.write('fbeta   {}\n'.format(fbeta))
            file.write('\n')
            file.write('\n')
        file.write('-------------')
        file.write('\n')

if __name__ == "__main__":   
    slice_performance(model, df, cat_features, encoder, lb)