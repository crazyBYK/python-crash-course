import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_feature_description():
    with open("../Data/Ames_Housing_Feature_Description.txt", "r") as f:
        print(f.read())


# df = pd.read_csv('../Data/Ames_NO_Missing_Data.csv')


def dealing_with_categorical_data(df: pd.DataFrame) -> pd.DataFrame:
    print(df.isnull().sum())
    df["MS SubClass"] = df["MS SubClass"].apply(str)
    my_object_df = df.select_dtypes(include="object")
    my_numeric_df = df.select_dtypes(exclude="object")
    df_object_dummies = pd.get_dummies(my_object_df, drop_first=True)
    final_df = pd.concat([my_numeric_df, df_object_dummies], axis=1)
    print(final_df.corr()["SalePrice"].sort_values())
    return final_df


if __name__ == "__main__":
    dealing_with_categorical_data()
