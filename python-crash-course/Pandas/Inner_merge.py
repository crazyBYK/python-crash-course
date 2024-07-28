import numpy as np
import pandas as pd

registration = pd.DataFrame(
    {"reg_id": [1, 2, 3, 4], "name": ["Andrew", "Bob", "Claire", "David"]}
)
login = pd.DataFrame(
    {"log_id": [1, 2, 3, 4], "name": ["Xavier", "Andrew", "Yolanda", "Bob"]}
)


def inner_merge():
    print(registration)
    print(login)
    users = pd.merge(registration, login, how="inner", on="name")
    print(users)
    cp_users = pd.merge(registration, login, how="inner", on="name")


def left_and_right_merge():
    left_join = pd.merge(left=registration, right=login, how="left", on="name")
    right_join = pd.merge(left=registration, right=login, how="right", on="name")


if __name__ == "__main__":
    # inner_merge()
    left_and_right_merge()
