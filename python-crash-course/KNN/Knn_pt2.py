# A Pipiline object in Scikit-Learn can set up a sequence of repeated operations, such as a scaler and a model.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.pipeline import Pipeline


def Knn_part2(df: pd.DataFrame):
    print(df.head())

    X = df.drop("Cancer Present", axis=1)
    y = df["Cancer Present"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    knn_model = KNeighborsClassifier(n_neighbors=1)
    knn_model.fit(scaled_X_train, y_train)

    y_pred = knn_model.predict(scaled_X_test)

    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    # print(accuracy_score(y_test, y_pred))

    test_error_rates = []

    for k in range(1, 30):
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(scaled_X_train, y_train)

        y_pred_test = knn_model.predict(scaled_X_test)

        test_error = 1 - accuracy_score(y_test, y_pred_test)

        test_error_rates.append(test_error)

    print(test_error_rates)

    plt.plot(range(1, 30), test_error_rates)
    plt.xlabel("K")
    plt.ylabel("test error rate")
    plt.show()


# Use Pipeline : gridsearch GV
def using_pipeline(df: pd.DataFrame):
    X = df.drop("Cancer Present", axis=1)
    y = df["Cancer Present"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()

    knn = KNeighborsClassifier()
    print(knn.get_params().keys())

    operations = [("scaler", scaler), ("knn", knn)]

    pipe = Pipeline(operations)

    K_values = list(range(1, 20))

    param_grid = {"knn__n_neighbors": K_values}

    full_cv_classifier = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy")

    full_cv_classifier.fit(X_train, y_train)

    print(full_cv_classifier.best_estimator_.get_params())

    full_prediction = full_cv_classifier.predict(X_test)

    print(classification_report(y_test, full_prediction))

    new_patient = [[3.8, 6.4]]

    print(full_cv_classifier.predict(new_patient))

    print(full_cv_classifier.predict_proba(new_patient))


if __name__ == "__main__":
    df = pd.read_csv("../Data/gene_expression.csv")
    print(df.describe())
    print(df.head())
    # Knn_part2(df)
    # using_pipeline(df)
