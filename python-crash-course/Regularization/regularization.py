import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv("Advertising.csv")
X = df.drop("sales", axis=1)
y = df["sales"]


def setup_data_ex(df):
    X = df.drop("sales", axis=1)
    y = df["sales"]
    polynomial_converter = PolynomialFeatures(degree=3, include_bias=False)
    poly_feature = polynomial_converter.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        poly_feature, y, test_size=0.3, random_state=101
    )

    # print(X_train.shape)

    scaler = StandardScaler()

    scaler.fit(X_train)
    # DO NOT USE test data set for it could be reason why data leakage

    scaled_X_train = scaler.transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    return scaled_X_test, scaled_X_train


if __name__ == "__main__":
    setup_data_ex()
