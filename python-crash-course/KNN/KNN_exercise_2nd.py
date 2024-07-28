import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def KNN_plot_exercise(df: pd.DataFrame):
    # sns.heatmap(df.corr(numeric_only=True))
    # plt.show()
    df["Target"] = df["Label"].map({"R": 0, "M": 1})
    sorted_corr = np.abs(df.corr(numeric_only=True)["Target"]).sort_values().tail(6)
    print(sorted_corr)


def KNN_exercise_part2(df: pd.DataFrame):
    print(df.head())
    X = df.drop("Label", axis=1)
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    scaler = StandardScaler()

    knn = KNeighborsClassifier()

    operations = [("scaler", scaler), ("knn", knn)]

    pipe = Pipeline(operations)

    K_values = list(range(1, 30))
    param_grid = {"knn__n_neighbors": K_values}

    full_cv_classifier = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy")

    full_cv_classifier.fit(X_train, y_train)

    print(full_cv_classifier.best_estimator_.get_params())

    mean_test_score = full_cv_classifier.cv_results_.get("mean_test_score")

    plt.plot(K_values, mean_test_score, "o-")
    plt.xlabel("K")
    plt.ylabel("mean test score")

    full_predictions = full_cv_classifier.predict(X_test)

    cl_report = classification_report(y_test, full_predictions)
    accuracy = accuracy_score(y_test, full_predictions)
    cf_metric = confusion_matrix(y_test, full_predictions)


if __name__ == "__main__":
    df = pd.read_csv("../Data/sonar.all-data.csv")
    # KNN_plot_exercise(df)
    KNN_exercise_part2(df)
