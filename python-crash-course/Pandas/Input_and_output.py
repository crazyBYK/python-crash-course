import pandas as pd
import os


def read_csv_file():
    df = pd.read_csv(
        "/Users/brian/Downloads/UNZIP_FOR_NOTEBOOKS_FINAL/03-Pandas/example.csv"
    )

    df.to_csv("output_csv.csv", index=False)
    print(df.head)


def read_html_table():
    url = "https://en.wikipedia.org/wiki/World_population"
    tables = pd.read_html(url)
    print(len(tables))
    print(tables[0].columns)
    world_population = tables[0]
    print(world_population)
    # world_pop = world_population.drop('#', axis=1)
    # print(world_pop)
    world_population.to_html("simple_table.html", index=False)


if __name__ == "__main__":
    # read_csv_file()
    read_html_table()
