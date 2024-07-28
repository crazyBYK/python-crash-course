import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from Regularization.regularization import setup_data_ex
from sklearn.linear_model import ElasticNetCV

df = pd.read_csv("Advertising.csv")


def data_preperation(data):
    X = data.drop("sales", axis=1)
    y = data["sales"]

    poly_converter = PolynomialFeatures(degree=3, include_bias=False)
    poly_feature = poly_converter.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        poly_feature, y, test_size=0.3, random_state=101
    )

    return X_train, X_test, y_train, y_test


def elastic_net():
    X_train, X_test, y_train, y_test = data_preperation(df)
    scaled_X_test, scaled_X_train = setup_data_ex(df)

    elastic_model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
        eps=0.001,
        n_alphas=100,
        max_iter=1000000,
    )
    elastic_model.fit(scaled_X_train, y_train)
    elastic_model_prediction = elastic_model.predict(scaled_X_test)

    elastic_MAE = mean_absolute_error(y_test, elastic_model_prediction)
    elastic_RESE = np.sqrt(mean_squared_error(y_test, elastic_model_prediction))

    print(elastic_model_prediction.l1_ratio_)
    print(elastic_MAE)
    print(elastic_RESE)


if __name__ == "__main__":
    elastic_net()
