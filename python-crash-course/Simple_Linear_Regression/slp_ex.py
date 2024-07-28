import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Advertising.csv")


def ex_code_for_ISLR():
    #     Simple Linear Regression
    #     y = ms + b
    #     print(df.head())
    df["total_spend"] = df["TV"] + df["radio"] + df["newspaper"]
    # print(df.head())
    # sns.scatterplot(data=df, x='total_spend', y='sales')
    # plt.show()
    # sns.regplot(data=df, x='total_spend', y='sales')
    # plt.show()

    X = df["total_spend"]
    y = df["sales"]

    # y = mx + b
    # y = B1x + B0
    # help(np.polyfit)
    regression = np.polyfit(X, y, deg=1)
    potential_spend = np.linspace(0, 500, 100)
    predicted_sales = regression[0] * potential_spend + regression[1]
    # sns.scatterplot(data=df, x='total_spend', y='sales')
    # plt.plot(potential_spend, predicted_sales, color='red')
    # plt.show()
    spend = 200
    predicted_sales = regression[0] * spend + regression[1]
    print(predicted_sales)

    deg3_regression = np.polyfit(X, y, deg=3)

    #     y = B1x + B0
    #     y = B3x**3 + B2x**2 + B1x + B0
    pot_spend = np.linspace(0, 500, 100)
    predicted_sales_3rd = (
        deg3_regression[0] * pot_spend**3
        + deg3_regression[1] * pot_spend**2
        + deg3_regression[2] * pot_spend
        + deg3_regression[3]
    )

    sns.scatterplot(data=df, x="total_spend", y="sales")
    plt.plot(pot_spend, predicted_sales_3rd, color="red")
    plt.show()


if __name__ == "__main__":
    # print(df.head())
    # df['total_spend'] = df['TV'] + df['radio'] + df['newspaper']
    # print(df.head())
    # sns.scatterplot(data=df, x='total_spend', y='sales')
    # sns.regplot(data=df, x='total_spend', y='sales')
    # plt.show()

    # X = df['total_spend']
    # y = df['sales']

    #   y = mx + b
    #   y = B1x +B0
    #   help(np.polyfit)

    # print(np.polyfit(X, y, deg=1))
    # potential_spend = np.linspace(0, 500, 100)
    # sns.scatterplot(data=df, x='total_spend', y='sales')
    # predicted_sales = 0.04868788 * potential_spend + 4.24302822
    # plt.plot(potential_spend, predicted_sales, color='red')
    # plt.show()
    # spend = 200
    # predicted_sales = 0.04868788 * spend + 4.24302822
    # print(predicted_sales)

    # y = B3*x**3 + B2*x**2 + B1*x + B0
    # print(np.polyfit(X, y, 3))
    # pot_spend = np.linspace(0, 500, 100)
    # pred_sales = 3.07615033e-07*pot_spend**3 + -1.89392449e-04*pot_spend**2 + 8.20886302e-02*pot_spend + 2.70495053e+00
    ex_code_for_ISLR()
