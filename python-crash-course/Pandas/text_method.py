import numpy as np
import pandas as pd
import timeit


def text_method():
    email = "jos@email.com"

    names = pd.Series(["andrew", "bobo", "claire", "david", "5"])
    print(names)
    print(names.str.isdigit())
    print(names.str.upper())
    tech_finance = ["GOOG,APPL,AMZN", "JPM,BAC,GS"]
    print(len(tech_finance))

    tickers = pd.Series(tech_finance)
    print(tickers)

    print(tickers.str.split(",").str[0])
    print(tickers.str.split(",", expand=True))  # str array -> Series -> Dataframe

    messy_name = pd.Series(["andrew ", "bo;bo", "  claire  "])
    print(messy_name)
    print(messy_name.str.replace(";", "").str.strip().str.capitalize())

    print(messy_name.apply(clearup))


def clearup(name: str):
    name = name.replace(";", "")
    name = name.strip()
    name = name.capitalize()
    return name


def efficient_check():
    setup = """
import pandas as pd
import numpy as np
messy_name = pd.Series(['andrew ', "bo;bo", "  claire  "])
    
def cleanup(name:str):
    name = name.replace(';','')
    name = name.strip()
    name = name.capitalize()
    return name
    """

    stmt_pandas_str = """
messy_name.str.replace(';','').str.strip().str.capitalize()
    """

    stmt_pandas_apply = """
messy_name = messy_name.apply(cleanup)
    """

    stmt_pandas_vectorize = """
np.vectorize(cleanup)(messy_name)
    """

    print(timeit.timeit(setup=setup, stmt=stmt_pandas_str, number=10000))
    print(timeit.timeit(setup=setup, stmt=stmt_pandas_apply, number=10000))
    print(timeit.timeit(setup=setup, stmt=stmt_pandas_vectorize, number=10000))


if __name__ == "__main__":
    # text_method()
    efficient_check()
