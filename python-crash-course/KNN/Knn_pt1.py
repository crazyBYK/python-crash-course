import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


def KNN_part01(df: pd.DataFrame):
    print(df.head())
    # sns.scatterplot(data=df, x='Gene One', y='Gene Two', hue='Cancer Present', alpha=0.6, style='Cancer Present')
    # plt.xlim(2,6)
    # plt.ylim(4,8)

    # sns.pairplot(data=df, hue='Cancer Present')
    # plt.show()

    X = df.drop("Cancer Present", axis=1)
    y = df["Cancer Present"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()

    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    knn_model = KNeighborsClassifier(n_neighbors=1)

    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    df = pd.read_csv("../Data/gene_expression.csv")
    KNN_part01(df)
