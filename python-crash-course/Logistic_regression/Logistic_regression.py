## Logistic Regression is a classification alogithm designed to predict categorical target labels.
## Logistic Regression will allow us to predict a catagorical label based on historical feature data.
## The categorical target column is two or more discrete class labels.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import PrecisionRecallDisplay


def check_data(df: pd.DataFrame):
    print(df.head())
    print(df.describe())
    print(df["test_result"].value_counts())

    fig = plt.figure(dpi=200)
    # sns.countplot(data=df, x='test_result')
    # sns.boxplot(data=df, x='test_result', y='physical_score')
    # sns.scatterplot(data=df, x='age', y='physical_score', hue='test_result', alpha=0.5)
    # sns.pairplot(data=df, hue='test_result')
    # sns.heatmap(df.corr(), annot=True)
    # sns.scatterplot(x='physical_score', y='test_result', data=df, alpha=0.5)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df["age"], df["physical_score"], df["test_result"])
    plt.show()


def logistic_regression(df: pd.DataFrame):
    X = df.drop("test_result", axis=1)
    y = df["test_result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=101
    )

    scaler = StandardScaler()

    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    log_model = LogisticRegression()
    log_model.fit(scaled_X_train, y_train)

    print(log_model.coef_)

    y_pred = log_model.predict(scaled_X_test)
    y_pred = log_model.predict_log_proba(scaled_X_test)
    y_pred = log_model.predict_proba(scaled_X_test)


def classification_metric_performance(df: pd.DataFrame):
    X = df.drop("test_result", axis=1)
    y = df["test_result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=101
    )

    scaler = StandardScaler()

    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    log_model = LogisticRegression()
    log_model.fit(scaled_X_train, y_train)

    y_pred = log_model.predict(scaled_X_test)
    ac_score = accuracy_score(y_test, y_pred)
    con_matrix = confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))

    prec_score = precision_score(y_test, y_pred)
    rec_score = recall_score(y_test, y_pred)

    print(prec_score)
    print(rec_score)

    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)

    disp.plot()

    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("../Data/hearing_test.csv")
    # check_data(df)
    # logistic_regression(df)
    classification_metric_performance(df)
