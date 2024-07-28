import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("Advertising.csv")


def create_polynomial_feature(X) -> df:
    polynomial_converter = PolynomialFeatures(degree=2, include_bias=False)
    polynomial_converter.fit(X)
    poly_feature = polynomial_converter.transform(X)
    return poly_feature


def polynomial_regression_ex():
    X = df.drop("sales", axis=1)
    y = df["sales"]

    poly_feature = create_polynomial_feature(X)

    print(poly_feature)

    X_train, X_test, y_train, y_test = train_test_split(
        poly_feature, y, test_size=0.3, random_state=101
    )

    print(X_train)
    model = LinearRegression()
    model.fit(X_train, y_train)

    test_prediction = model.predict(X_test)
    print(model.coef_)

    MAE = mean_absolute_error(y_test, test_prediction)
    # print(MAE)

    RMSE = np.sqrt(mean_squared_error(y_test, test_prediction))
    # print(RMSE)

    print(poly_feature[0])


if __name__ == "__main__":
    # create_polynomial_feature()
    polynomial_regression_ex()
