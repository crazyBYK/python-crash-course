import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_ages(mu=50, sigma=13, num_sample=100, seed=42) -> [int]:
    np.random.seed(seed)
    sample_ages = np.random.normal(loc=mu, scale=sigma, size=num_sample)
    sample_ages = np.round(sample_ages, decimals=0)
    return sample_ages


def trimming_domain_data():
    sample = create_ages()
    ser = pd.Series(sample)
    ser.apply(fix_values)
    # print_distplot(sample)
    q25, q75 = np.percentile(sample, [25, 75])
    iqr = q75 - q25
    print(iqr)


def print_distplot(data):
    sns.displot(data=data, bins=20)
    plt.show()


def fix_values(age):
    if age < 18:
        return 18
    else:
        return age


def dealing_with_outliner_with_AMES_DATA():
    df = pd.read_csv("../Data/Ames_Housing_Data.csv")
    sale_rel = df.corr(numeric_only=True)["SalePrice"].sort_values()
    print(sale_rel)
    outliner = df[(df["Overall Qual"] > 8) & (df["SalePrice"] < 200000)]
    print(outliner)
    print_scatterplot(df)
    drop_index = df[(df["Gr Liv Area"] > 4000) & (df["SalePrice"] < 400000)].index
    prepare_df = df.drop(drop_index, axis=0)
    print_scatterplot(prepare_df)
    save_file(prepare_df)


def save_file(data):
    data.to_csv("../Own_data/Ames_outliners_removed_Data.csv")


def print_scatterplot(data):
    sns.scatterplot(x="Overall Qual", y="SalePrice", data=data)
    plt.show()


if __name__ == "__main__":
    # trimming_domain_data()
    dealing_with_outliner_with_AMES_DATA()
