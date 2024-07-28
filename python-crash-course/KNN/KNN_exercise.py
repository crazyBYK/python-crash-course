import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def KNN_exercise(df: pd.DataFrame):
    print(df.head())
    # df_data = df.drop('Label', axis=1)
    # print(df_data_2.head())
    # print(df_data_2.nlargest(n=5, columns='Freq_1'))
    # plt.show()
    # sns.heatmap(df.corr(numeric_only=True))

    # converting label into numaric value as Integer.
    df["Target"] = df["Label"].map({"R": 0, "M": 1})
    # print(df['Target'])
    # print(df.head())
    print(np.abs(df.corr(numeric_only=True)["Target"]).sort_values().tail(6))


def KNN_exercise_part2(df: pd.DataFrame):
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
    # param_grid = {'knn_n_neighbors': K_values}

    full_cv_classifier = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy")

    full_cv_classifier.fit(X_train, y_train)

    print(full_cv_classifier.best_estimator_.get_params())

    mean_test_score = full_cv_classifier.cv_results_.get("mean_test_score")

    print(mean_test_score)

    plt.plot(range(1, 30), mean_test_score, "o-")
    plt.xlabel("K")
    plt.ylabel("Mean Test Score")
    plt.show()

    full_prediction = full_cv_classifier.predict(X_test)

    cl_report = classification_report(y_test, full_prediction)

    accuracy = accuracy_score(y_test, full_prediction)

    cf_metric = confusion_matrix(y_test, full_prediction)

    print(cl_report)
    print(accuracy)
    print(cf_metric)


if __name__ == "__main__":
    df = pd.read_csv("../Data/sonar.all-data.csv")
    print(df.head())

    # KNN_exercise(df)
    # KNN_exercise_part2(df)
