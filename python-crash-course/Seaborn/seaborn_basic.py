import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dm_office_sales.csv")


def seaborn_01():
    plt.figure(figsize=(12, 4), dpi=200)
    sns.scatterplot(
        x="salary", y="sales", data=df, hue="level of education", palette="Dark2"
    )
    plt.show()


def seaborn_type_02():
    plt.figure(figsize=(12, 4), dpi=200)
    sns.scatterplot(
        x="salary",
        y="sales",
        data=df,
        s=200,
        style="level of education",
        hue="level of education",
    )
    plt.savefig("my_plt.png")
    plt.show()


if __name__ == "__main__":
    # seaborn_01()
    seaborn_type_02()
