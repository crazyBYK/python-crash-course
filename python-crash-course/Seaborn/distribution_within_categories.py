import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("StudentsPerformance.csv")


def ex_boxplot():
    # plt.figure(figsize=(10, 4), dpi=200)
    plt.figure(figsize=(10, 8), dpi=200)
    # sns.boxplot(data=df, y='math score', x='test preparation course')
    # sns.boxplot(data=df, y='reading score', x='parental level of education', hue='test preparation course')
    sns.boxplot(
        data=df,
        x="reading score",
        y="parental level of education",
        hue="test preparation course",
        palette="Set2",
    )
    plt.legend(bbox_to_anchor=(1.05, 0.5))
    plt.show()


def ex_violinplot():
    plt.figure(figsize=(10, 8), dpi=200)
    # sns.violinplot(data=df, x='reading score', y='parental level of education')
    # sns.violinplot(data=df, x='reading score', y='parental level of education', hue='test preparation course', palette='Set2', split=True, inner=None)

    sns.violinplot(
        data=df, x="reading score", y="parental level of education", bw_method=0.01
    )

    # error "No artists with labels found to put in legend => because we don't have hue data
    # plt.legend(bbox_to_anchor=(1.25, 0.5))
    plt.show()


def ex_swarmplot():
    plt.figure(figsize=(10, 8), dpi=200)
    sns.swarmplot(
        data=df, x="math score", y="gender", size=2, hue="test preparation course"
    )
    plt.legend(bbox_to_anchor=(1.05, 0.5))
    plt.show()


def ex_boxenplot():
    sns.boxenplot(data=df, x="math score", y="test preparation course", hue="gender")
    plt.show()


if __name__ == "__main__":
    # ex_boxplot()
    # ex_violinplot()
    # ex_swarmplot()
    ex_boxenplot()
