# The cross_val_score function uses a model and training set(along with a K and
# chosen metric) to perform all of this for us automatically.
# This allows for K-Fold cross validation to be performed on any model.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score


def func_cross_validation(df: pd.DataFrame):
    X = df.drop("sales", axis=1)
    y = df["sales"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=101
    )
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = Ridge(alpha=100)

    scores = cross_val_score(
        model, X_train, y_train, scoring="neg_mean_squared_error", cv=5
    )

    print(scores)

    abs(scores.mean())

    model = Ridge(alpha=1)

    scores_second = cross_val_score(
        model, X_train, y_train, scoring="neg_mean_squared_error", cv=5
    )
    abs(scores_second.mean())

    model.fit(X_train, y_train)

    y_final_test_pred = model.predict(X_test)

    final_eval_scores = mean_squared_error(y_test, y_final_test_pred)

    print(final_eval_scores)


if __name__ == "__main__":
    df = pd.read_csv("../Data/Advertising.csv")
    func_cross_validation(df)
