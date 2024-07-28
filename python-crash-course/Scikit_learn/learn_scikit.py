import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy as sp
from joblib import dump, load

df = pd.read_csv("Advertising.csv")


def subplot_advertising_method():
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))

    axes[0].plot(df["TV"], df["sales"], "o")
    axes[0].set_ylabel("Sales")
    axes[0].set_title("TV Spend")

    axes[1].plot(df["radio"], df["sales"], "o")
    axes[1].set_ylabel("Sales")
    axes[1].set_title("Radio Spend")

    axes[2].plot(df["newspaper"], df["sales"], "o")
    axes[2].set_title("Newspaper Spend")
    axes[2].set_ylabel("Sales")

    plt.tight_layout()
    plt.show()


def using_pairplot():
    sns.pairplot(df)
    plt.show()


def using_linear_regression():
    X = df.drop("sales", axis=1)
    y = df["sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101
    )

    # model = LinearRegression()
    # model.fit(X_train, y_train)  #return LinearRegression()
    # print(model.predict(X_test))

    model = LinearRegression()
    model.fit(X_train, y_train)


def regression_metrics():
    X = df.drop("sales", axis=1)
    y = df["sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # print(model.predict(X_test)) # y_test
    # print(X_test.head())
    # print(y_test.head())

    test_predictions = model.predict(X_test)
    print(df["sales"].mean())
    # sns.histplot(data=df, x='sales', bins=20)
    # plt.show()

    mae = mean_absolute_error(y_test, test_predictions)
    print(mae)

    mes = mean_squared_error(y_test, test_predictions)
    rmes = np.sqrt(mes)
    print(mes)
    print(rmes)

    test_residuals = y_test - test_predictions
    print(test_residuals)
    #   test_residual
    #     sns.scatterplot(x=y_test, y=test_residuals)
    #     plt.axhline(y=0, color='r', ls='--')
    #     plt.show()
    #     sns.displot(test_residuals, bins=25, kde=True)
    #     plt.show()

    fig, ax = plt.subplots(figsize=(6, 8), dpi=100)
    _ = sp.stats.probplot(test_residuals, plot=ax)
    plt.show()


def deployments():
    X = df.drop("sales", axis=1)
    y = df["sales"]

    final_model = LinearRegression()
    final_model.fit(X, y)
    print(final_model.coef_)
    print(X.head())

    y_hat = final_model.predict(X)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))

    axes[0].plot(df["TV"], df["sales"], "o")
    axes[0].plot(df["TV"], y_hat, "o", color="red")
    axes[0].set_ylabel("Sales")
    axes[0].set_xlabel("TV Spend")

    axes[1].plot(df["radio"], df["sales"], "o")
    axes[1].plot(df["radio"], y_hat, "o", color="red")
    axes[1].set_ylabel("Sales")
    axes[1].set_xlabel("Radio Spend")

    axes[2].plot(df["newspaper"], df["sales"], "o")
    axes[2].plot(df["newspaper"], y_hat, "o", color="red")
    axes[2].set_ylabel("Sales")
    axes[2].set_xlabel("newspaper Spend")

    plt.tight_layout()
    plt.show()


def save_load():
    X = df.drop("sales", axis=1)
    y = df["sales"]

    final_model = LinearRegression()
    final_model.fit(X, y)

    # dump(final_model, 'final_sales_model.joblib')

    loaded_model = load("final_sales_model.joblib")
    print(loaded_model.coef_)

    print(X.shape)

    # 149 TV, 22 Radio, 12 Newspaper
    # Sales?
    campaign = [[149, 22, 12]]
    campaign_predict = loaded_model.predict(campaign)
    print(campaign_predict)


if __name__ == "__main__":
    # print(df.head())
    # subplot_advertising_method()
    # using_pairplot()

    # X = df.drop('sales', axis=1)
    # X
    # y = df['sales']
    # help(train_test_split)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    # print(len(df))
    # print(len(X_train))

    # using_linear_regression()
    # regression_metrics()
    deployments()
    # save_load()
