import pandas as pd
import numpy as np
from Regularization.regularization import setup_data_ex
from sklearn.linear_model import Ridge, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Important Note!!
# SKlearn refers to lambda as alpha within the class call.

df = pd.read_csv("Advertising.csv")


def data_preparation(data):
    X = data.drop("sales", axis=1)
    y = data["sales"]

    poly_converter = PolynomialFeatures(degree=3, include_bias=False)
    ploy_feature = poly_converter.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        ploy_feature, y, test_size=0.3, random_state=101
    )

    return X_train, X_test, y_train, y_test


def ridge_regression():
    X_train, X_test, y_train, y_test = data_preparation(df)
    scaled_X_test, scaled_X_train = setup_data_ex(df)

    ridge_model = Ridge(alpha=10)
    ridge_model.fit(scaled_X_train, y_train)

    test_predictions = ridge_model.predict(scaled_X_test)

    MAE = mean_absolute_error(y_test, test_predictions)
    RMSE = np.sqrt(mean_squared_error(y_test, test_predictions))

    print(MAE)
    print(RMSE)

    ridge_cv_model = RidgeCV(alphas=(0.1, 1.0, 10.0), scoring="neg_mean_absolute_error")
    ridge_cv_model.fit(scaled_X_train, y_train)
    test_cv_predictions = ridge_cv_model.predict(scaled_X_test)
    cv_MAE = mean_absolute_error(y_test, test_cv_predictions)
    cv_RMSE = np.sqrt(mean_squared_error(y_test, test_cv_predictions))

    print(cv_MAE)
    print(cv_RMSE)

    print(ridge_cv_model.coef_)


if __name__ == "__main__":
    ridge_regression()
