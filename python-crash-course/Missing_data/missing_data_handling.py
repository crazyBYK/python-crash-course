import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("../Data/Ames_outliers_removed.csv")
pd.options.mode.copy_on_write = True


def missing_data_handling():
    # with open("../Data/Ames_Housing_Feature_Description.txt", "r") as f:
    #     print(f.read())

    # print(df.info())
    df_v1 = df.drop("PID", axis=1)
    # print(df_missing)
    df_missing: pd.DataFrame = percent_missing(df_v1)
    sns.barplot(x=df_missing.index, y=df_missing)
    # plt.show()
    df_v2 = df_v1.dropna(axis=0, subset=["Electrical", "Garage Cars"])
    df_v2_missing: df.DataFrame = percent_missing(df_v2)
    sns.barplot(x=df_v2_missing.index, y=df_v2_missing)

    bsmt_num_cols = [
        "BsmtFin SF 1",
        "BsmtFin SF 2",
        "Bsmt Unf SF",
        "Total Bsmt SF",
        "Bsmt Full Bath",
        "Bsmt Half Bath",
    ]

    df_v2[bsmt_num_cols] = df_v2[bsmt_num_cols].fillna(0)
    bsmt_str_cols = [
        "Bsmt Qual",
        "Bsmt Cond",
        "Bsmt Exposure",
        "BsmtFin Type 1",
        "BsmtFin Type 2",
    ]
    df_v2[bsmt_str_cols] = df_v2[bsmt_str_cols].fillna("None")
    print(df_v2.head())
    print(df_v2.info())
    print(df_v2[df_v2["Bsmt Full Bath"].isnull()])

    data_filled_na: df.DataFrame = percent_missing(df_v2)
    sns.barplot(x=data_filled_na.index, y=data_filled_na)
    plt.xticks(rotation=90)
    # plt.ylim(0, 1)
    plt.show()


def missing_data_handling_2():
    df_v1 = df.drop("PID", axis=1)
    df_v1_missing: pd.DataFrame = percent_missing(df_v1)
    df_v2 = df_v1.dropna(axis=0, subset=["Electrical", "Garage Cars"])
    df_v2_missing: df.DataFrame = percent_missing(df_v2)
    bsmt_num_cols = [
        "BsmtFin SF 1",
        "BsmtFin SF 2",
        "Bsmt Unf SF",
        "Total Bsmt SF",
        "Bsmt Full Bath",
        "Bsmt Half Bath",
    ]
    df_v2[bsmt_num_cols] = df_v2[bsmt_num_cols].fillna(0)
    bsmt_str_cols = [
        "Bsmt Qual",
        "Bsmt Cond",
        "Bsmt Exposure",
        "BsmtFin Type 1",
        "BsmtFin Type 2",
    ]
    df_v2[bsmt_str_cols] = df_v2[bsmt_str_cols].fillna("None")
    print(df_v2)

    data_filled_na: df.DataFrame = percent_missing(df_v2)
    # sns.barplot(x=data_filled_na.index, y=data_filled_na)
    # plt.xticks(rotation=90)
    # plt.show()
    # print(df_v2.info())


def percent_missing(df: pd.DataFrame) -> pd.DataFrame:
    percent_nan = 100 * df.isnull().sum() / len(df)
    percent_nan = percent_nan[percent_nan > 0].sort_values()
    return percent_nan


if __name__ == "__main__":
    # missing_data_handling()

    # with open("../Data/Ames_Housing_Feature_Description.txt", "r") as f:
    #     print(f.read())

    # with open("../Data/Ames_Housing_Feature_Description.txt", "r") as f:
    #     print(f.read())
    missing_data_handling_2()
