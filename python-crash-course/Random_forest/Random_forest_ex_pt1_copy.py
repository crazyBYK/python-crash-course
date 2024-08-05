import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def random_forest_copy(df: pd.DataFrame):
    # print(df.head())
    # sns.pairplot(data=df, hue='Class')
    # plt.show()

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=101
    )

    n_estimators = [64, 100, 128, 200]
    max_features = [2, 3, 4]
    bootstrap = [True, False]
    oob_score = [True, False]

    param_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "bootstrap": bootstrap,
        "oob_score": oob_score,
    }

    model_for_parameter = RandomForestClassifier()
    grid = GridSearchCV(model_for_parameter, param_grid)

    # grid.fit(X_train, y_train)
    #
    # print(grid.best_params_)

    model_for_fit = RandomForestClassifier(
        max_features=2, n_estimators=200, oob_score=True, bootstrap=True
    )

    model_for_fit.fit(X_train, y_train)

    pred = model_for_fit.predict(X_test)

    cfm = confusion_matrix(y_test, pred)

    print(cfm)

    errors = []
    misClassificaitons = []

    for n in range(0, 100):
        rfc = RandomForestClassifier(n_estimators=n, max_features=2)
        rfc.fit(X_train, y_train)
        preds = rfc.predict(X_test)
        err = 1 - accuracy_score(y_test, preds)
        n_missed = np.sum(preds != y_test)

        errors.append(err)
        misClassificaitons.append(n_missed)

    plt.plot(range(0, 100), misClassificaitons)
    plt.show()


    pass


if __name__ == "__main__":
    df = pd.read_csv("../Data/data_banknote_authentication.csv")
    random_forest_copy(df)
