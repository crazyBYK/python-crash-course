# The cross_validation function allows us to view multiple performance metrics from
# cross validation on a model and explore how much time fitting and testing took.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler


def cross_validation_func(df: pd.DataFrame):
    ## create X and y
    X = df.drop("sales", axis=1)
    y = df["sales"]

    ## train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101
    )

    ## scale data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = Ridge(alpha=100)

    scores = cross_validate(
        model,
        X_train,
        y_train,
        scoring=["neg_mean_squared_error", "neg_mean_absolute_error"],
        cv=10,
    )
    scores = pd.DataFrame(scores)
    print(scores)
    print(scores.mean())

    model = Ridge(alpha=1)

    scores = cross_validate(
        model,
        X_train,
        y_train,
        scoring=["neg_mean_squared_error", "neg_mean_absolute_error"],
        cv=10,
    )

    model.fit(X_train, y_train)

    model_pred = model.predict(X_test)

    model_evaluate_score = mean_squared_error(y_test, model_pred)

    print(model_evaluate_score)


if __name__ == "__main__":
    df = pd.read_csv("../Data/Advertising.csv")
    cross_validation_func(df)
