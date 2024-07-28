import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error


def project_solution(df: pd.DataFrame):
    print(df.head())

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=101
    )

    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    # base_elastic_model = ElasticNet()
    base_elastic_model = ElasticNet(max_iter=1000000)

    param_grid = {"alpha": [0.1, 1, 5, 10, 100], "l1_ratio": [0.1, 0.7, 0.99, 1]}

    grid_model = GridSearchCV(
        estimator=base_elastic_model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        verbose=1,
    )
    grid_model.fit(scaled_X_train, y_train)
    print(grid_model.best_params_)
    print(grid_model.best_estimator_)

    y_pred = grid_model.predict(scaled_X_test)
    MAE_score = mean_absolute_error(y_test, y_pred)
    RMSE_score = np.sqrt(mean_squared_error(y_test, y_pred))

    print(MAE_score)
    print(RMSE_score)


if __name__ == "__main__":
    df = pd.read_csv("../Data/AMES_Final_DF.csv")
    project_solution(df)
