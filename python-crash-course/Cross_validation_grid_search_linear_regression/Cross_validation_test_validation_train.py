import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def test_validation_train(df: pd.DataFrame):
    print(df.head())
    X = df.drop("sales", axis=1)
    y = df["sales"]
    # creating test
    X_train, X_other, y_train, y_other = train_test_split(
        X, y, test_size=0.3, random_state=101
    )
    # test_size = 0.5 (50% of 30% other ----> test = 15% of all data)
    # creating validation dataframe for adjust training configure set
    X_eval, X_test, y_eval, y_test = train_test_split(
        X_other, y_other, test_size=0.5, random_state=101
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_eval = scaler.transform(X_eval)
    X_test = scaler.transform(X_test)

    model_one = Ridge(alpha=100)
    model_one.fit(X_train, y_train)

    y_eval_pred = model_one.predict(X_eval)

    model_one_evaluate_score = mean_squared_error(y_eval, y_eval_pred)
    print(model_one_evaluate_score)

    model_two = Ridge(alpha=1)
    model_two.fit(X_train, y_train)

    y_eval_pred_two = model_two.predict(X_eval)

    model_two_evaluate_score = mean_squared_error(y_eval, y_eval_pred_two)
    print(model_two_evaluate_score)

    # once finish adjust model. then using test dataset for final evaluation.
    y_test_pred_final = model_two.predict(X_test)

    y_test_evaluate_score = mean_squared_error(y_test, y_test_pred_final)
    print(y_test_evaluate_score)


if __name__ == "__main__":
    df = pd.read_csv("../Data/Advertising.csv")
    test_validation_train(df)
