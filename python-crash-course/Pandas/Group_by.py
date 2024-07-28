import numpy as np
import pandas as pd


def group_by_part01():
    df = pd.read_csv("mpg.csv")
    print(df)
    print(df["model_year"].unique())
    print(df["model_year"].value_counts())
    # df.groupby('model_year')
    # print(df.groupby('model_year').mean())

    print(df.head())
    print(df.groupby("model_year"))
    print(df.groupby("model_year").mean(numeric_only=True))
    print(df.groupby("model_year").mean(numeric_only=True)["mpg"])

    print(df.groupby(["model_year", "cylinders"]).mean(numeric_only=True))
    print(df.groupby(["model_year", "cylinders"]).mean(numeric_only=True).index)
    print(df.groupby("model_year").describe().transpose())

    year_cyl = df.groupby(["model_year", "cylinders"]).mean(numeric_only=True)
    # from panda 2.0.0., changed default value of numeric_only from True to False, so it have to have parameter for it.

    print(year_cyl)
    print(year_cyl.index.names)
    print(year_cyl.index.levels)

    print(year_cyl.loc[[70, 82]])
    print(year_cyl.loc[(70, 4)])


def group_by_part02():
    df = pd.read_csv("mpg.csv")

    print(df)
    year_cyl = df.groupby(["model_year", "cylinders"]).mean(numeric_only=True)
    print(year_cyl)
    print(year_cyl.index.levels)
    print(year_cyl.xs(key=70, level="model_year"))
    print(year_cyl.loc[[70, 80]])

    print(year_cyl.xs(key=4, level="cylinders"))
    print(year_cyl)
    six_and_eight = (
        df[df["cylinders"].isin([6, 8])]
        .groupby(["model_year", "cylinders"])
        .mean(numeric_only=True)
    )
    print(six_and_eight)

    print(year_cyl.swaplevel())

    print(year_cyl.sort_index(level="model_year", ascending=False))
    test_agg = df.agg({"mpg": ["max", "mean"], "weight": ["mean", "std"]})
    print(test_agg)


if __name__ == "__main__":
    group_by_part02()
