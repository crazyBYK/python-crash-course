import numpy as np
import pandas as pd

if __name__ == "__main__":
    np.random.seed(101)
    mydata = np.random.randint(0, 101, (4, 3))
    myindex = ["CA", "NY", "AZ", "TX"]
    mycolumn = ["Jan", "Feb", "Mar"]

    df = pd.DataFrame(mydata, myindex, mycolumn)
    print(df)
    print(df.info)

    df_tips = pd.read_csv("tips.csv")
    print(df_tips)

    print(df_tips.columns)
    print(df_tips.index)
    print(df_tips.head(10))
    print(df_tips.tail(10))
    print(df_tips.info())
    print(df_tips.describe())

    print(df_tips["total_bill"])
    print(type(df_tips["total_bill"]))
    print(df_tips[["total_bill", "tip"]])
    myColumn = ["total_bill", "tip"]
    print(df_tips[myColumn])

    sum_columns = 100 * df_tips["tip"] / df_tips["total_bill"]

    print(sum_columns)
    df_tips["tip_percentage"] = 100 * df_tips["tip"] / df_tips["total_bill"]
    print(df_tips.head())
    df_tips["price_per_person"] = df_tips["total_bill"] / df_tips["size"]
    df_tips["price_per_person"] = np.round(df_tips["total_bill"] / df_tips["size"], 2)
    print(df_tips["price_per_person"])

    df_tips.drop("tip_percentage", axis=1)
    df_tips = df_tips.drop("tip_percentage", axis=1)
    print(df_tips.head())
    print(df_tips.shape[1])
    print(df_tips.index)
    df_tips = df_tips.set_index("Payment ID")
    print(df_tips.head)
    df_tips = df_tips.reset_index()

    df_tips = df_tips.set_index("Payment ID")
    print(df_tips.iloc[0])
    print(df_tips.loc["Sun2959"])
    print(df_tips.iloc[0:4])
    print(df_tips.loc[["Sun2959", "Sun4608"]])
    df = df_tips.drop("Sun2959", axis=0)
    print(df.head)

    one_row = df.iloc[0]
    print(one_row)
    df.append(one_row)
