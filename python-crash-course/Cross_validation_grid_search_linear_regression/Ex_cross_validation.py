import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Train | Test Split Procedure.
# 0. Clean and adjust data as necessary for X and y
# 1. Split Data in Train/Test for both X and y
# 2. Fit/Train Scaler on Training X and y
# 3. Scale X Test Data
# 4. Create Model
# 5. Fit/Train Model on X Train Data
# 6. Evaluate Model on X Test Data (by creating predictions and comparing to Y_test)
# 7. Adjust Parameters as Necessary and repeat steps 5 and 6


def ex_cross_validation(df: pd.DataFrame, alpha: int):
    print(df.head())
    X = df.drop("sales", axis=1)
    y = df["sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101
    )

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    evaluate_score = mean_squared_error(y_test, y_pred)
    print(evaluate_score)


if __name__ == "__main__":
    df = pd.read_csv("../Data/Advertising.csv")
    ex_cross_validation(df, 1)
