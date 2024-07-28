import pandas as pd
import numpy as np

df = pd.read_csv(
    "/Users/brian/Downloads/UNZIP_FOR_NOTEBOOKS_FINAL/03-Pandas/Sales_Funnel_CRM.csv"
)
print(df)
# print(df.columns)
# print(help(pd.pivot))


def pivot_ex():
    licenses = df[["Company", "Product", "Licenses"]]
    print(licenses.head())
    # licenses_pivot = pd.pivot(data=licenses, index='Company', columns='Product', values='Licenses')
    licenses_pivot = pd.pivot(
        data=licenses, index="Company", columns="Product", values="Licenses"
    )
    # print(licenses_pivot.head())


def pivot_table_ex():
    numeric = df[["Company", "Account Number", "Licenses", "Sale Price"]]
    google_account = pd.pivot_table(
        numeric, index="Company", aggfunc="sum", values=["Licenses", "Sale Price"]
    )
    # google_account = pd.pivot_table(df, index='Company', aggfunc='sum')
    # print(google_account.columns)
    # print(google_account)

    # sales_price_by_contact = pd.pivot_table(df, index=['Account Manager', 'Contact'], values=['Sale Price'], columns=['Product'], aggfunc='sum', fill_value=0)
    # sales_price_by_contact = pd.pivot_table(df, index=['Account Manager', 'Contact'], values=['Sale Price'], columns=['Product'], aggfunc=[np.sum, np.mean], fill_value=0)
    sales_price_by_contact = pd.pivot_table(
        df,
        index=["Account Manager", "Contact", "Product"],
        values=["Sale Price"],
        columns=["Product"],
        aggfunc=[np.sum],
        fill_value=0,
        margins=True,
    )

    print(sales_price_by_contact)


def group_by_ex():
    groupby_sample = df.groupby("Company").sum(numeric_only=True)
    print(groupby_sample)


if __name__ == "__main__":
    # pivot_ex()
    pivot_table_ex()
    # group_by_ex()
