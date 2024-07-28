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
    print(len(ser.apply(fix_values)))


def fix_values(age):
    if age < 18:
        return 18
    else:
        return age


def dealing_with_outliner():
    sample = create_ages()

    # sns.displot(sample, bins=20)
    # plt.show()
    #
    # sns.boxplot(sample)
    # plt.show()

    ser = pd.Series(sample)
    print(ser.describe())

    # easy to get Q1 and Q3 value by using np.percentile
    q25, q75 = np.percentile(sample, [25, 75])


def dealing_with_outliner_with_AMES_DATA():
    df = pd.read_csv("../Data/Ames_Housing_Data.csv")
    df.head()
    df.corr(numeric_only=True)
    df.corr()["SalePrice"].sort_values()


if __name__ == "__main__":
    trimming_domain_data()
