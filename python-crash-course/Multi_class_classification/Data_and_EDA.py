import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def data_and_eda(df: pd.DataFrame):
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df["species"].value_counts())

    # sns.countplot(x='species', data=df)
    # sns.scatterplot(x ='petal_length', y='petal_width', data=df, hue='species')
    # sns.pairplot(data=df, hue='species')
    # sns.heatmap(df.corr(), annot=True)
    # plt.show()
    X = df.drop("species", axis=1)
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=101
    )
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    log_model = LogisticRegression(solver="saga", multi_class="ovr", max_iter=5000)

    penalty = ["l1", "l2", "elasticnet"]
    l1_ratio = np.linspace(0, 1, 20)
    C = np.linspace(0, 10, 20)

    param_grid = {"penalty": penalty, "l1_ratio": l1_ratio, "C": C}

    grid_model = GridSearchCV(log_model, param_grid=param_grid)
    grid_model.fit(scaled_X_train, y_train)

    print(grid_model.best_params_)

    y_pred = grid_model.predict(scaled_X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(report)


if __name__ == "__main__":
    df = pd.read_csv("../Data/iris.csv")
    data_and_eda(df)
