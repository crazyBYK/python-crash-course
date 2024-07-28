import numpy as np
import pandas as pd

registration = pd.DataFrame(
    {"reg_id": [1, 2, 3, 4], "name": ["Andrew", "Bob", "Claire", "David"]}
)
login = pd.DataFrame(
    {"log_id": [1, 2, 3, 4], "name": ["Xavier", "Andrew", "Yolanda", "Bob"]}
)


def outer_merge():
    outer_merge = pd.merge(registration, login, how="outer", on="name")
    print(outer_merge)


def merge_with_index():
    registrations = registration.set_index("name")
    left_index_merge = pd.merge(
        registrations, login, left_index=True, right_on="name", how="inner"
    )
    print(left_index_merge)


def merge_with_diff_name():
    print(registration)
    print(login)
    registration.columns = ["reg_id", "reg_name"]
    print(registration)
    result = pd.merge(
        registration, login, how="inner", left_on="reg_name", right_on="name"
    )
    print(result)
    result1 = result.drop("reg_name", axis=1)
    print(result1)

    registration.columns = ["id", "name"]
    login.columns = ["id", "name"]

    print(registration)
    print(login)

    result2 = pd.merge(
        registration, login, how="inner", on="name", suffixes=("_reg", "_log")
    )
    print(result2)


if __name__ == "__main__":
    # outer_merge()
    # merge_with_index()
    merge_with_diff_name()
