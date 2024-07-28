import numpy as np
import pandas as pd

if __name__ == "__main__":
    # help(pd.Series)
    myindex = ["USA", "Canada", "Maxico"]
    mydata = [1776, 1867, 1821]
    myser1 = pd.Series(data=mydata)
    myser = pd.Series(data=mydata, index=myindex)
    print(myser)
    print(myser1[0])
    print(myser["USA"])

    ages = {"Sam": 5, "Frank": 10, "Spike": 7}
    myAges = pd.Series(ages)
    print(myAges)

    q1 = {"Japan": 80, "China": 450, "India": 200, "USA": 250}
    q2 = {"Brazil": 100, "China": 500, "India": 210, "USA": 260}

    sales_q1 = pd.Series(q1)
    sales_q2 = pd.Series(q2)

    print(sales_q1.iloc[0])
    print(sales_q1["Japan"])

    print(sales_q1.keys())

    print(sales_q1 * 2)
    print(sales_q1 / 100)

    print(sales_q1 * 100)
    sales_q1 = pd.Series(q1)
    sales_q2 = pd.Series(q2)
    print(sales_q1 + sales_q2)

    sales_add = sales_q1.add(sales_q2, fill_value=0)
    print(sales_add.dtype)

    df_tips = pd.read_csv("tips.csv")
    print(df_tips)
