import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from missing_data_handling import percent_missing as pm


df = pd.read_csv("../Data/Ames_outliers_removed.csv")


def missing_data_fixing_data_based_on_column():
    df_v1 = df.drop("PID", axis=1)
    df_v1_missing = pm(df_v1)
    df_v2 = df_v1.dropna(axis=0, subset=["Electrical", "Garage Cars"])

    bsmt_num_cols = [
        "BsmtFin SF 1",
        "BsmtFin SF 2",
        "Bsmt Unf SF",
        "Total Bsmt SF",
        "Bsmt Full Bath",
        "Bsmt Half Bath",
    ]
    bsmt_str_cols = [
        "Bsmt Qual",
        "Bsmt Cond",
        "Bsmt Exposure",
        "BsmtFin Type 1",
        "BsmtFin Type 2",
    ]

    df_v2[bsmt_num_cols] = df_v2[bsmt_num_cols].fillna(0)
    df_v2[bsmt_str_cols] = df_v2[bsmt_str_cols].fillna("None")

    gar_str_cols = ["Garage Type", "Garage Finish", "Garage Qual", "Garage Cond"]
    gar_num_cols = ["Garage Yr Blt"]

    df_v2[gar_str_cols] = df_v2[gar_str_cols].fillna("None")
    df_v2[gar_num_cols] = df_v2[gar_num_cols].fillna(0)
    df_v2_misssing = pm(df_v2)
    # sns.barplot(x=df_v2_misssing.index, y=df_v2_misssing)

    df_v3 = df_v2.drop(axis=1, columns=["Pool QC", "Misc Feature", "Alley", "Fence"])
    df_v3_missing: df.DataFrame = pm(df_v3)
    # sns.barplot(x=df_v3_missing.index, y=df_v3_missing)
    # plt.xticks(rotation=90)
    # plt.show()
    # print(df_v3['Fireplace Qu'].value_counts())
    df_v3["Fireplace Qu"] = df_v3["Fireplace Qu"].fillna("None")
    df_v3_missing: df.DataFrame = pm(df_v3)

    # plt.figure(figsize=(8, 12))
    # sns.barplot(x=df_v3_missing.index, y=df_v3_missing)
    # plt.xticks(rotation=90)
    # plt.show()

    df_v3["Mas Vnr Area"] = df_v3["Mas Vnr Area"].fillna(0)
    df_v4 = df_v3.drop(axis=1, columns=["Mas Vnr Type"])
    df_v4_missing: df.DataFrame = pm(df_v3)
    plt.figure(figsize=(8, 12))
    sns.barplot(x=df_v4_missing.index, y=df_v4_missing)
    # plt.xticks(rotation=90)
    # plt.show()
    # sns.boxplot(x='Lot Frontage', y='Neighborhood', data=df_v3, orient='h')
    # plt.show()

    df_v4["Lot Frontage"] = df_v4.groupby("Neighborhood")["Lot Frontage"].transform(
        lambda value: value.fillna(value.mean())
    )

    print(df_v4.isnull().sum())

    df_v4["Lot Frontage"] = df_v4["Lot Frontage"].fillna(0)

    print(df_v4.isnull().sum())


if __name__ == "__main__":
    # with open('../Data/Ames_Housing_Feature_Description.txt') as f:
    #     print(f.read())

    print("loading data...")
    missing_data_fixing_data_based_on_column()
