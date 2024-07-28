import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


def knn_model_grid(df: pd.DataFrame):
    # Assuming the last column is the target variable
    # X = df.iloc[:, :-1]
    # y = df.iloc[:, -1]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # Create a pipeline
    pipeline = Pipeline([
        ('knn', KNeighborsClassifier())
    ])

    # Create space of candidate values
    search_space = {
        'knn__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }

    # Create grid search
    clf = GridSearchCV(pipeline, search_space, cv=5, verbose=0)

    # Fit grid search
    best_model = clf.fit(X_train, y_train)

    # View best model
    best_k = best_model.best_estimator_.get_params()['knn__n_neighbors']
    print(f'Best number of neighbors: {best_k}')

    # Predict the responses for test dataset
    y_pred = best_model.predict(X_test)

    return y_pred


if __name__ == '__main__':
    df = pd.read_csv('../Data/sonar.all-data.csv')
    print(df.head())