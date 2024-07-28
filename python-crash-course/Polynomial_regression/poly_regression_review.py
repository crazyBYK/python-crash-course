import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import dump, load

df = pd.read_csv("Advertising.csv")


def reshape_dataset():
    X = df.drop("sales", axis=1)
    y = df["sales"]
    return X, y


def create_poly_feature(X):
    polynomial_converter = PolynomialFeatures(degree=2, include_bias=False)
    ploy_feature = polynomial_converter.fit_transform(X)
    return ploy_feature


def evaluating_poly_feature_degree():
    train_rmse_error = []
    test_rmse_error = []

    X, y = reshape_dataset()

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

    plt.plot(range(1, 6), train_rmse_error[:5], label="train rmse")
    plt.plot(range(1, 6), test_rmse_error[:5], label="test rmse")
    plt.ylabel("RMSE")
    plt.xlabel("Degree of Poly")
    plt.legend()
    plt.show()

    res_converter = PolynomialFeatures(degree=3, include_bias=False)
    res_converted_x = res_converter.fit_transform(X)

    res_model = LinearRegression()
    res_model.fit(res_converted_x, y)

    return res_converter, res_model


def deploy_selected_poly_model():
    final_converter, final_model = evaluating_poly_feature_degree()
    dump(final_converter, "final_poly_converter_review.joblib")
    dump(final_model, "final_poly_model_review.joblib")


def load_selected_poly_model():
    deploy_selected_poly_model()

    loaded_converter = load("final_poly_converter_review.joblib")
    loaded_model = load("final_poly_model_review.joblib")

    campaign = [[149, 22, 12]]

    transformed_data = loaded_converter.fit_transform(campaign)
    campaign_prediction = loaded_model.predict(transformed_data)
    print(campaign_prediction)

    pass


if __name__ == "__main__":
    # evaluating_poly_feature_degree()
    load_selected_poly_model()
