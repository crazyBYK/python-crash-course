import numpy as np
import pandas as pd


def concat():
    data_one = {"A": ["A0", "A1", "A2", "A3"], "B": ["B0", "B1", "B2", "B3"]}
    data_two = {"C": ["C0", "C1", "C2", "C3"], "D": ["D0", "D1", "D2", "D3"]}

    one = pd.DataFrame(data_one)
    two = pd.DataFrame(data_two)

    print(one)
    print(two)

    join = pd.concat([one, two], axis=1)
    print(join)

    print(pd.concat([one, two], axis=0))
    two.columns = one.columns
    mydf = pd.concat([one, two], axis=0)
    print(mydf)
    mydf.index = range(len(mydf))
    print(mydf)


if __name__ == "__main__":
    concat()
