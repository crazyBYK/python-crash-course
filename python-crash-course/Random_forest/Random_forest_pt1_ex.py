import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def random_forest_pt1_ex(df: pd.DataFrame):
    print(df.head())
    sns.countplot(data=df, x='class')
    plt.show()

    print(df.describe())
    pass



if __name__ == '__main__':
    df = pd.read_csv('../Data/mushrooms.csv')
    random_forest_pt1_ex(df)