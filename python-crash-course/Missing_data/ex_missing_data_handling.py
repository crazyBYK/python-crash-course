import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("../Data/Ames_outliers_removed.csv")


def ex_missing_data_handling():
    df_v1 = df.drop("PID", axis=1)
    df_missing: df.DataFrame = percent_missing(df_v1)
    # print(df_missing)
    sns.barplot(x=df_missing.index, y=df_missing)
    plt.xticks(rotation=90)
    plt.ylim(0, 1)
    # plt.show()
    # print(df_missing[df_missing < 1])
    # print(df[df['Electrical'].isnull()]['Garage Area'])
    df_v2 = df.drop("PID", axis=1)
    df_v2 = df_v2.dropna(axis=0, subset=["Electrical", "Garage Cars"])
    # print(df_v2)
    df_v2_missing: df.DataFrame = percent_missing(df_v2)
    sns.barplot(x=df_v2_missing.index, y=df_v2_missing)
    plt.xticks(rotation=90)
    plt.ylim(0, 1)
    # plt.show()

    # BSMT NUMERIC COLUMNS --> fillna 0
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
    plt.ylim(0, 1)
    plt.show()


def percent_missing(df: pd.DataFrame) -> pd.DataFrame:
    percent_nan = 100 * df.isnull().sum() / len(df)
    percent_nan = percent_nan[percent_nan > 0].sort_values()
    return percent_nan


if __name__ == "__main__":
    ex_missing_data_handling()
