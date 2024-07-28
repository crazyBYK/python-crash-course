# A grid search is a way of training and validating a model on every possible combination of multiple hyperparameter options

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler


def test_grid_search(df: pd.DataFrame):
    print(df.head())
    ## create X and y
    X = df.drop("sales", axis=1)
    y = df["sales"]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101
    )

    # scale data
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    base_elastic_net_model = ElasticNet()
    param_grid = {
        "alpha": [0.1, 1, 5, 10, 50, 100],
        "l1_ratio": [0.1, 0.5, 0.7, 0.95, 0.99, 1],
    }

    grid_model = GridSearchCV(
        estimator=base_elastic_net_model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        verbose=2,
    )

    grid_model.fit(X_train, y_train)

    print(grid_model.best_params_)
    print(grid_model.best_estimator_)
    print(grid_model.cv_results_)
    grid_results = pd.DataFrame(grid_model.cv_results_)
    print(grid_results)

    y_pred = grid_model.predict(X_test)
    result = mean_squared_error(y_test, y_pred)
    print(result)


if __name__ == "__main__":
    df = pd.read_csv("../Data/Advertising.csv")
    test_grid_search(df)
