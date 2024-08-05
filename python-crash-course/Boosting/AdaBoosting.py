import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def adaBooting_ex(df: pd.DataFrame):
    print(df.head())
    print(df.describe().transpose().reset_index().sort_values('unique'))
    feat_uni = df.describe().transpose().reset_index().sort_values('unique')
    # plt.figure(figsize=(14, 6), dpi=200)
    # sns.barplot(data=feat_uni, x='index', y='unique')
    # plt.xticks(rotation=90)
    # plt.show()

    X = df.drop('class', axis=1)
    X = pd.get_dummies(X, drop_first=True)

    y = df['class']

#   TODO : split train and test dataset



if __name__ == '__main__':
    df = pd.read_csv("../Data/mushrooms.csv")
    adaBooting_ex(df)