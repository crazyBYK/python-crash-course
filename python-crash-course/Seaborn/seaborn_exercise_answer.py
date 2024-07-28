import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("application_record.csv")


def task_one():
    employed = df[df["DAYS_EMPLOYED"] < 0]
    employed["DAYS_EMPLOYED"] = employed["DAYS_EMPLOYED"] * -1
    employed["DAYS_BIRTH"] = employed["DAYS_BIRTH"] * -1
    print(employed[["DAYS_EMPLOYED", "DAYS_BIRTH"]])
    plt.figure(figsize=(12, 6), dpi=200)
    sns.scatterplot(
        data=employed, x="DAYS_BIRTH", y="DAYS_EMPLOYED", alpha=0.01, linewidth=0
    )
    plt.show()


def task_two():
    df["YEARS"] = -1 * df["DAYS_BIRTH"] / 365
    sns.histplot(data=df, x="YEARS", bins=45, color="red")
    plt.show()


def task_three():
    bottom_half_income = df.nsmallest(n=int(len(df) / 2), columns="AMT_INCOME_TOTAL")
    sns.boxplot(
        data=bottom_half_income,
        y="AMT_INCOME_TOTAL",
        x="NAME_FAMILY_STATUS",
        hue="FLAG_OWN_REALTY",
    )
    plt.figure(figsize=(12, 5), dpi=200)
    plt.legend(loc=(1.05, 0.5), title="FLAG_OWN_REALTY")
    plt.show()


def task_four():
    sns.heatmap(df.drop("FLAG_MOBIL", axis=1).corr(numeric_only=True), cmap="viridis")
    plt.show()


if __name__ == "__main__":
    # task_one()
    # task_two()
    # task_three()
    task_four()
