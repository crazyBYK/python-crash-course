import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def adjusting_polynomial_feature_degree():
    # create the different order poly
    # split poly feature train/test
    # fit on train
    # store/save the rmse for BOTH the train aND test
    # PLOT the results (error vs poly order)
    train_rmse_error = []
    test_rmse_error = []

    X = df.drop("sales", axis=1)
    y = df["sales"]

    for d in range(1, 10):
        poly_converter = PolynomialFeatures(degree=d, include_bias=False)
        poly_feature = poly_converter.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            poly_feature, y, test_size=0.3, random_state=101
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))

        train_rmse_error.append(train_rmse)
        test_rmse_error.append(test_rmse)

    print(train_rmse_error)
    print(test_rmse_error)

    plt.plot(range(1, 6), train_rmse_error[:5], label="TRAIN RMSE")
    plt.plot(range(1, 6), test_rmse_error[:5], label="TEST RMSE")
    plt.ylabel("RMSE")
    plt.xlabel("Degree of Poly")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    adjusting_polynomial_feature_degree()
