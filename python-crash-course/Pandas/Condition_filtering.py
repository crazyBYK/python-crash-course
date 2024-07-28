import numpy as np
import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("tips.csv")
    print(df.head())

    print(df["total_bill"] > 40)
    bool_series = df["total_bill"] > 40
    print(df[bool_series])
    print(df[df["total_bill"] > 40])

    print(df[df["sex"] == "Male"])

    print(df[df["size"] > 3])

    #     AND & --- BOTH CONDITIONS NEED TO BE TRUE
    #     OR | --- EITHER CONDITIONS IS TRUE

    print(df[(df["total_bill"] > 30) & (df["sex"] == "Male")])
    print(df[(df["total_bill"] < 30) | (df["sex"] == "Male")])
    print(df[(df["day"] == "Sun") | (df["day"] == "Sat")])

    conditions = ["Sun", "Sat"]
    print(df[df["day"].isin(conditions)])
    print(df[df["day"].isin(["Sun", "Sat"])])
