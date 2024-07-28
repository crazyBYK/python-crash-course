import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

df = pd.read_csv("../Data/Advertising.csv")


# Train | Test Split Procedure
# 0. Clean and adjust data as necessary for X and y
# 1. Split Data in Train/Test for both X and y
# 2. Fit/Train Scaler on Training X and y
# 3. Scale X Test Data
# 4. Create Model
# 5. Fit/Train Model on X Train Data
# 6. Evaluate Model on X Test Data (by creating predictions and comparing to Y_test)
# 7. Adjust Parameters as Necessary and repeat steps 5 and 6


def cross_validation():
    print(df.head())
    #     0. Clear and adjust data as necessary for X and y
    #     1. Split Data in Train/Test for both X and y
    X = df.drop("sales", axis=1)
    y = df["sales"]

    #     2. Fit/Train Scaler on Training X and y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101
    )

    #     3. Scale X Test Data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #     4. Create Model
    #     5. Fit/Train Model on X Train Data
    model = Ridge(alpha=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #     6. Evaluate Model on X Test Data (by creating predictions and comparing to Y_test)
    evaluate_score = mean_squared_error(y_test, y_pred)
    print(evaluate_score)

    #     7. Adjust Parameters as Necessary and repeat steps 5 and 6
    model_two = Ridge(alpha=1)
    model_two.fit(X_train, y_train)
    y_pred_two = model_two.predict(X_test)
    evaluate_score_two = mean_squared_error(y_test, y_pred_two)
    print(evaluate_score_two)


if __name__ == "__main__":
    cross_validation()
