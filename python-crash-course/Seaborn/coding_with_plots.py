import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("StudentsPerformance.csv")


def ex_jointplot():
    # sns.jointplot(data=df, x='math score', y='reading score', kind='hex')
    # sns.jointplot(data=df, x='math score', y='reading score', kind='scatter', alpha=0.2)
    # sns.jointplot(data=df, x='math score', y='reading score', kind='hist')
    # sns.jointplot(data=df, x='math score', y='reading score', kind='kde', shade=True)
    sns.jointplot(data=df, x="math score", y="reading score", hue="gender")
    plt.show()


def ex_pairplot():
    sns.pairplot(data=df, hue="gender", corner=True)
    plt.show()


def ex_catplot():
    sns.catplot(
        data=df,
        x="gender",
        y="math score",
        kind="box",
        col="lunch",
        row="race/ethnicity",
    )
    plt.show()


if __name__ == "__main__":
    print(df.head())
    # ex_jointplot()
    # ex_pairplot()
    ex_catplot()
