import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error

from Regularization.regularization import setup_data_ex

df = pd.read_csv("Advertising.csv")


def data_preperation(data):
    X = data.drop("sales", axis=1)
    y = data["sales"]

    poly_converter = PolynomialFeatures(degree=3, include_bias=False)
    ploy_feature = poly_converter.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        ploy_feature, y, test_size=0.3, random_state=101
    )

    return X_train, X_test, y_train, y_test


def lasso_regression():
    # least absolute shrinkage and selection operator
    X_train, X_test, y_train, y_test = data_preperation(df)
    scaled_X_test, scaled_X_train = setup_data_ex(df)

    lasso_cv_model = LassoCV(eps=0.0001, n_alphas=100, cv=5, max_iter=1000000)
    lasso_cv_model.fit(scaled_X_train, y_train)

    lasso_test_prediction = lasso_cv_model.predict(scaled_X_test)
    lasso_MAE = mean_absolute_error(y_test, lasso_test_prediction)
    lasso_RMSE = np.sqrt(mean_squared_error(y_test, lasso_test_prediction))

    print(lasso_MAE)
    print(lasso_RMSE)

    print(lasso_cv_model.coef_)


if __name__ == "__main__":
    lasso_regression()
