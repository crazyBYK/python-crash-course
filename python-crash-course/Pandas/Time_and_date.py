import numpy as np
import pandas as pd

from datetime import datetime


def date_and_time_python():
    myyear = 2015
    mymonth = 1
    myday = 1
    myhour = 2
    mymin = 30
    mysec = 15

    mydate = datetime(myyear, mymonth, myday)
    mydateetime = datetime(myyear, mymonth, myday, myhour, mymin, mysec)

    print(mydate)
    print(mydateetime)


def date_and_time_pandas():
    myser = pd.Series(["Nov 3, 1990", "2000-01-01", None])
    timeser = pd.to_datetime(myser, format="mixed")
    print(timeser)
    print(timeser[0].year)
    print(timeser[0])

    obvi_euro_date = "31-12-2000"
    print(pd.to_datetime(obvi_euro_date, dayfirst=True))

    style_date = "12--Dec--2000"

    print(pd.to_datetime(style_date, format="%d--%b--%Y"))

    custom_date = "12th of Dec 2000"
    print(pd.to_datetime(custom_date))

    sales = pd.read_csv("RetailSales_BeerWineLiquor.csv")
    print(sales)
    sales["DATE"] = pd.to_datetime(sales["DATE"])
    print(sales["DATE"])

    sales2 = pd.read_csv(
        "RetailSales_BeerWineLiquor.csv", parse_dates=[0]
    )  # automatically parse date column
    print(sales2.head)

    sales2 = sales2.set_index("DATE")  # set index by DATE
    print(sales2.resample(rule="A").mean())  # group by Rule
    print(sales["DATE"].dt.month)  # dt stands for datetime


if __name__ == "__main__":
    # date_and_time_python()
    date_and_time_pandas()
