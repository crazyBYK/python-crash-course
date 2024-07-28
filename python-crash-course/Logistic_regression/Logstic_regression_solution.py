import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix, classification_report


def ex_solution(df: pd.DataFrame):
    print(df.head())
    print(df.info())
    print(df.isnull().sum())
    print(df.describe())

    plt.figure(figsize=(12, 8), dpi=300)
    sns.countplot(x="target", data=df)
    print(df.columns)
    sns.pairplot(df[["age", "trestbps", "chol", "thalach", "target"]], hue="target")

    sns.heatmap(df.corr(), annot=True, cmap="viridis")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=101
    )
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    log_model = LogisticRegressionCV()
    log_model.fit(scaled_X_train, y_train)

    print(log_model.Cs_)

    print(log_model.get_params())

    print(log_model.coef_)

    coefs = pd.Series(index=X.columns, data=log_model.coef_[0])
    coefs = coefs.sort_values()
    sns.barplot(x=coefs.index, y=coefs.values)

    y_pred = log_model.predict(scaled_X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    patient = [[54.0, 1.0, 0.0, 122.0, 286.0, 0.0, 0.0, 116.0, 1.0, 3.2, 1.0, 2.0, 2.0]]

    log_model.predict(patient)
    log_model.predict_proba(patient)


if __name__ == "__main__":
    df = pd.read_csv("../Data/heart.csv")
    ex_solution(df)
