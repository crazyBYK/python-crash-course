import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("dm_office_sales.csv")


def statistical_estimation():
    print(df["division"].value_counts())
    plt.figure(figsize=(10, 4), dpi=300)
    sns.countplot(data=df, x="division")
    plt.ylim(90, 260)
    plt.show()


def statistical_estimation_02():
    plt.figure(figsize=(10, 4), dpi=300)
    print(df["level of education"].value_counts())
    # sns.countplot(data=df, x='level of education')
    sns.countplot(data=df, x="level of education", hue="division", palette="Set2")
    plt.show()


if __name__ == "__main__":
    # statistical_estimation()
    statistical_estimation_02()
