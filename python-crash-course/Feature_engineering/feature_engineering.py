import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("../Data/Ames_Housing_Data.csv")


if __name__ == "__main__":
    print(df.head)
