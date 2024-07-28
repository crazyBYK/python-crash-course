import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from svm_margin_plot import plot_svm_boundary
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
)


def svm_ex(df: pd.DataFrame):
    # print(df.head())
    sns.scatterplot(df, x="Med_1_mL", y="Med_2_mL", hue="Virus Present")
    # plt.show()

    # HYPERPLANE (2d line)
    x = np.linspace(0, 10, 100)
    m = -1
    b = 11
    y = m * x + b

    plt.plot(x, y, "black")
    # plt.show()

    # help(SVC)

    y = df["Virus Present"]
    X = df.drop("Virus Present", axis=1)
    model = SVC(kernel="linear", C=1000)
    model.fit(X, y)

    plot_svm_boundary(model, X, y)


def svm_part2(df: pd.DataFrame):
    plt.figure(figsize=(10, 10), dpi=300)
    sns.heatmap(df.corr(numeric_only=True), annot=True)
    # plt.show()
    print(df.columns)
    X = df.drop("Compressive Strength (28-day)(Mpa)", axis=1)
    y = df["Compressive Strength (28-day)(Mpa)"]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    # scaler = StandardScaler()
    # scaled_X_train = scaler.fit_transform(X_train, y_train)
    # scaled_X_test = scaler.transform(X_test)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101
    )
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    # help(SVR)

    base_model = SVR()

    base_model.fit(scaled_X_train, y_train)

    base_predict = base_model.predict(scaled_X_test)

    mae = mean_absolute_error(y_test, base_predict)
    print(mae)

    mse = np.sqrt(mean_squared_error(y_test, base_predict))
    print(mse)

    print(y_test.mean())

    param_grid = {
        "C": [0.001, 0.01, 0.1, 0.5, 1],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"],
        "degree": [2, 3, 4],
        "epsilon": [0, 0.01, 0.02, 0.1, 0.5, 1, 2],
    }

    svr = SVR()

    grid = GridSearchCV(svr, param_grid)

    grid.fit(scaled_X_train, y_train)

    # print(grid.best_params_)

    grid_predict = grid.predict(scaled_X_test)

    grid_mae = mean_absolute_error(y_test, grid_predict)
    print(grid_mae)

    grid_rmse = np.sqrt(mean_squared_error(y_test, grid_predict))
    print(grid_rmse)

    pass


if __name__ == "__main__":
    df = pd.read_csv("../Data/mouse_viral_study.csv")
    df2 = pd.read_csv("../Data/cement_slump.csv")
    # svm_ex(df)
    svm_part2(df2)
