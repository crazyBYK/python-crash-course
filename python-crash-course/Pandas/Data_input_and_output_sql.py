import pandas as pd
import numpy as np

from sqlalchemy import create_engine


temp_db = create_engine("sqlite:///:memory:")


def data_output_sql():
    df = pd.DataFrame(
        data=np.random.randint(low=0, high=100, size=(4, 4)),
        columns=["A", "B", "C", "D"],
    )
    df.to_sql(name="new_table", con=temp_db, if_exists="append", index=False)


def date_input_sql():
    df = pd.read_sql(sql="new_table", con=temp_db)
    print(df)
    df_sql = pd.read_sql_query(sql="SELECT a,c FROM new_table", con=temp_db)
    print(df_sql)


if __name__ == "__main__":
    data_output_sql()
    date_input_sql()
