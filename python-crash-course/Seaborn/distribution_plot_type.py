import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dm_office_sales.csv")


def distribution_type_01():
    plt.figure(figsize=(5, 8), dpi=200)
    sns.rugplot(x="salary", data=df, height=0.5)
    plt.show()


def distribution_type_02():
    plt.figure(figsize=(5, 8), dpi=200)
    # DO NOT USE DISTPLOT # DEPRECATED!!
    # sns.set(style='white')
    # sns.displot(data=df, x='salary', bins=20, color='red', edgecolor='blue', linewidth=0.5, ls='--')
    # sns.displot(data=df, x='salary', bins=10)
    sns.displot(data=df, x="salary", kde=True, rug=True)
    plt.show()


def distribution_type_03():
    plt.figure(figsize=(5, 8), dpi=200)
    # sns.histplot(data=df, x='salary')
    sns.kdeplot(data=df, x="salary")
    plt.show()


def distribution_type_04():
    # plt.figure(figsize=(5,8), dpi=200)
    np.random.seed(42)
    sample_ages = np.random.randint(0, 100, 200)
    sample_ages = pd.DataFrame(sample_ages, columns=["age"])
    # sns.rugplot(data=sample_ages, x='age')
    # sns.displot(data=sample_ages, x='age', rug=True, bins=30, kde=True)
    sns.kdeplot(data=sample_ages, x="age", clip=[0, 100], bw_adjust=0.01, shade=True)
    plt.show()


if __name__ == "__main__":
    # distribution_type_01()
    # distribution_type_02()
    # distribution_type_03()
    distribution_type_04()
