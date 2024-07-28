import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import PrecisionRecallDisplay


def exercise(df: pd.DataFrame):
    # print(df.head())
    # print(df['target'].unique())
    # print(df['target'].value_counts)
    # print(df.info())
    # print(df.describe())
    df_dex = df.describe().transpose()
    # sns.countplot(x='target', data=df)
    plt.figure(dpi=300)
    # sns.pairplot(data=df, vars=['age', 'trestbps', 'chol', 'thalach'], hue='target')
    sns.heatmap(df.corr(), annot=True)
    plt.show()


def exercise_pt1(df: pd.DataFrame):
    print(df.info())
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=101
    )
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    # help(LogisticRegressionCV)

    # log_model = LogisticRegression(solver='saga', multi_class='ovr', max_iter=5000)
    penalty = ["l1", "l2", "elasticnet"]
    l1_ratio = np.linspace(0, 1, 20)
    C = np.linspace(0, 10, 20)
    param_grid = {"C": C, "penalty": penalty, "l1_ratio": l1_ratio}

    # log_model = LogisticRegression(solver='saga', multi_class='ovr', max_iter=5000)
    #
    # grid_model = GridSearchCV(log_model, param_grid=param_grid)
    # grid_model.fit(scaled_X_train, y_train)
    #
    # print(grid_model.best_params_)
    #
    # y_pred = grid_model.predict(scaled_X_test)
    #
    # accuracy = accuracy_score(y_test, y_pred)
    # conf_matrix = confusion_matrix(y_test, y_pred)
    # report = classification_report(y_test, y_pred)

    # TODO : implement into LogisticRegressionCV (error occurred)
    log_model = LogisticRegressionCV().fit(scaled_X_train, y_train)

    # print(log_model.Cs_)
    # print(log_model.coef_)
    # print(log_model.get_params())

    # coefs = pd.Series(index=X.columns, data=log_model.coef_[0])
    # coefs = coefs.sort_values()
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x=coefs.index, y=coefs.values)
    # plt.show()

    y_pred = log_model.predict(scaled_X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    PrecisionRecallDisplay.from_estimator(log_model, scaled_X_test, y_test)
    # PrecisionRecallDisplay.from_predictions(y_test, y_pred)
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("../Data/heart.csv")
    # exercise(df)
    exercise_pt1(df)
