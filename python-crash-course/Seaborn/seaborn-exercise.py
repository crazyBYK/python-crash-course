import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


df = pd.read_csv("application_record.csv")
pd.set_option("display.max_columns", 3000)
pd.set_option("display.max_rows", 3000)
pd.options.display.width = None
pd.options.display.max_columns = None


def task_01():
    # Recreate the Scatter Plot shown below
    # The scatterplot attempts to show the relationship between the days employed versus the age of
    # person (DAYS_BRITH) for people who were not unemployed.
    warnings.simplefilter("ignore")
    plt.figure(figsize=(12, 8))

    # REMOVE UNEMPLOYED PEOPLE
    employed = df[df["DAYS_EMPLOYED"] < 0]

    # MAKE BOTH POSITIVE
    employed["DAYS_EMPLOYED"] = -1 * employed["DAYS_EMPLOYED"]
    employed["DAYS_BIRTH"] = -1 * employed["DAYS_BIRTH"]

    sns.scatterplot(
        data=employed, y="DAYS_EMPLOYED", x="DAYS_BIRTH", alpha=0.01, linewidth=0
    )
    plt.savefig("task_one.jpg")


def task_02():
    #     Recreate the Distribution plot shown below:
    #     Note, you will need to figure out how to calculate "Age in Years" from one of the columns
    #     in the DF. Think carefully about this. Don't worry too much if you are unable to replicate the styling exactly
    df["YEARS"] = -1 * df["DAYS_BIRTH"] / 365
    plt.figure(figsize=(8, 4))
    sns.histplot(
        data=df,
        x="YEARS",
        linewidth=2,
        edgecolor="black",
        color="red",
        bins=45,
        alpha=0.4,
    )
    plt.xlabel("Age in YEARS")
    plt.savefig("task_two.jpg")


def task_03():
    # Recreate the Categorical Plot show below:
    # This plot shows information only for bottom half of income earners in the data set.
    # It showns the boxplots for each category of NAME_FAMILY_STATUS column for displaying distribution
    # of their total income.
    # The hue is the 'FLAG_OWN_REALTY' column.

    plt.figure(figsize=(12, 5))
    bottom_half_income = df.nsmallest(n=int(0.5 * len(df)), columns="AMT_INCOME_TOTAL")
    sns.boxplot(
        data=bottom_half_income,
        x="NAME_FAMILY_STATUS",
        y="AMT_INCOME_TOTAL",
        hue="FLAG_OWN_REALTY",
    )
    plt.legend(
        bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, title="FLAG_OWN_REALTY"
    )
    plt.title("Income Totals per Family Status for Bottom Half of Earners")

    plt.savefig("task_three.jpg")


def task_04():
    # Recreate the Heat Map show below:
    # This heatmap shows the correlation between column in the dataframe. You cna get correlation with .corr(),
    # also note that the FLAG_MOBIL column has NaN correlation with every other column, so you should drop it before calling .corr()
    print(df.corr(numeric_only=True))
    sns.heatmap(df.drop("FLAG_MOBIL", axis=1).corr(numeric_only=True), cmap="viridis")
    plt.savefig("task_four.jpg")


if __name__ == "__main__":
    # task_01()
    # task_02()
    # task_03()
    task_04()
