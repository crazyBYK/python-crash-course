import numpy as np
import pandas as pd
import timeit

df = pd.read_csv("tips.csv")


def last_four(num: int) -> str:
    return str(num)[-4:]


def yelp(price: float) -> str:
    if price < 10:
        return "$"
    elif 10 <= price < 30:
        return "$$"
    else:
        return "$$$"


def simple(num: int) -> int:
    return num * 2


def quality(total_bill: float, tip: float) -> str:
    if tip / total_bill > 0.25:
        return "Generous"
    else:
        return "Other"


if __name__ == "__main__":
    # print(df.head())
    # print(df.info())
    # print(last_four(1234567))
    # print(df["CC Number"].apply(last_four))
    # df["last_four"] = df["CC Number"].apply(last_four)
    # df['yelp'] = df['total_bill'].apply(yelp)
    # print(df['total_bill'].apply(lambda num : num * 2))
    # print(df[['total_bill', 'tip']].apply(lambda df: quality(df['total_bill'], df['tip']), axis=1))
    # print(df[['total_bill', 'tip']].apply(lambda df: quality(df['total_bill'], df['tip']), axis=1))
    # df['Quality'] = df[['total_bill', 'tip']].apply(lambda df: quality(df['total_bill'], df['tip']), axis=1)
    # print(df)
    # print(np.vectorize(quality)(df['total_bill'],df['tip']))
    #  df['Quality2'] = np.vectorize(quality)(df['total_bill'], df['tip'])
    #
    # df['Quality2'] = np.vectorize(quality)(df['total_bill'], df['tip'])
    # print(df.head())

    setup = """
import numpy as np
import pandas as pd
df = pd.read_csv('tips.csv')
def quality(total_bill, tip):
    if tip/total_bill > 0.25:
        return 'Generous'
    else :
        return 'Other'
    """

    stmt_one = """df['Tip Quality'] = df[['total_bill', 'tip']].apply(lambda df : quality(df['total_bill'], df['tip']), axis=1)"""
    stmt_two = (
        """df['Tip Quality'] = np.vectorize(quality)(df['total_bill'], df['tip'])"""
    )

    # print(timeit.timeit(setup=setup, stmt=stmt_one, number=1000))
    # print(timeit.timeit(setup=setup, stmt=stmt_two, number=1000))

    df = pd.read_csv("tips.csv")

    print(df.describe())
    print(df.sort_values("tip"))
    print(df.sort_values(["tip", "size"]))
    print(df["total_bill"].max())  # finding max value of selected column
    print(
        df["total_bill"].idxmax()
    )  # finding index value of max value of selected column
    print(df.iloc[170])
    print(df.iloc[df["total_bill"].idxmin()])
    print(df["sex"].value_counts())
    print(df["day"].unique())
    print(df["day"].nunique())
    print(df["day"].value_counts())
    print(df["sex"].replace({["Female", "Male"], ["F", "M"]}))
    print(df.head())
    mymap = {"Female": "F", "Male": "M"}
    print()
