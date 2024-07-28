import numpy as np
import pandas as pd


if __name__ == "__main__":
    # df = pd.read_csv('tips.csv')
    # print(df.head())
    print(np.nan)
    print(pd.NA)
    print(pd.NaT)
    print(np.nan == np.nan)  # => False
    print(np.nan is np.nan)  # => True
    myvar = np.nan
    print(myvar is np.nan)  # => True

    df = pd.read_csv("movie_scores.csv")
    print(df.head())

    print(df.iloc[1])

    print(df.isnull())
    print(df.notnull())
    print(df["pre_movie_score"].notnull())
    print(df[df["pre_movie_score"].notnull()])

    print(df[(df["pre_movie_score"].isnull()) & (df["first_name"].notnull())])
    print(df[(df["pre_movie_score"].isnull()) & (df["first_name"].notnull())])

    # KEEP DATA
    # DROP DATA
    help(df.dropna())
    print(
        df.dropna(thresh=1)
    )  # thresh:int, optional => Require that many non-NA values. Cannot be combined with how.
    print(df.dropna(thresh=5))
    print(df.dropna(axis=1))
    print(df.dropna(subset=["last_name"]))

    print(df.fillna("NEW VALUE!!"))

    # print(df['pre_movie_score'].fillna(0))
    # df['pre_movie_score'] = df['pre_movie_score'].fillna(0)
    # print(df)

    print(df["age"].mean())

    airline_tix = {"first": 100, "business": np.nan, "economy-plus": 50, "economy": 30}
    ser = pd.Series(airline_tix)
    print(ser)
    print(ser.interpolate())

    # FILL DATA
