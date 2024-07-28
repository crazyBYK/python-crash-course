import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def ex2_dealing_with_outliner_with_data():
    df = pd.read_csv("../Data/Ames_Housing_Data.csv")
    sale_rel = df.corr(numeric_only=True)["SalePrice"].sort_values()
    print(sale_rel)
    outliner = df[(df["Overall Qual"] > 8) & (df["SalePrice"] < 200000)]
    print(outliner)
    drop_index = df[(df["Gr Liv Area"] > 4000) & (df["SalePrice"] < 400000)].index
    print(drop_index)
    prepare_df = df.drop(drop_index, axis=0)
    ex_print_scatterplot(prepare_df)
    save_file(prepare_df)


def ex_print_scatterplot(data):
    sns.scatterplot(x="Overall Qual", y="SalePrice", data=data)
    plt.show()


def save_file(data):
    data.to_csv("../Own_data/Ames_outliners_removed_data.csv")


if __name__ == "__main__":
    ex2_dealing_with_outliner_with_data()
