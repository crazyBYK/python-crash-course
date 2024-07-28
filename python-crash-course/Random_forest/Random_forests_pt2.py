import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def random_forest_pt2(df: pd.DataFrame):
    print(df.head())
    # sns.pairplot(df, hue='Class')
    # plt.show()

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)

    n_estimators = [64, 100, 128, 200]
    max_features = [2, 3, 4]
    bootstrap = [True, False]
    oob_score = [True, False]

    param_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'bootstrap': bootstrap, 'oob_score': oob_score}

    rfc = RandomForestClassifier()

    grid = GridSearchCV(rfc, param_grid)

    # grid.fit(X_train, y_train)

    # print(grid.best_params_)

    rfc = RandomForestClassifier(max_features=2, n_estimators=100, oob_score=True, bootstrap=True )

    rfc.fit(X_train, y_train)

    print(rfc.oob_score_)

    predictions = rfc.predict(X_test)

    cfm = confusion_matrix(y_test, predictions)

    print(cfm)

    cl_report = classification_report(y_test, predictions)

    print(cl_report)

    errors = []
    misClassifications = []

    for n in range(1, 100):
        rfc = RandomForestClassifier(n_estimators=n, max_features=2)
        rfc.fit(X_train, y_train)
        preds = rfc.predict(X_test)
        err = 1 - accuracy_score(y_test, preds)
        n_missed = np.sum(preds != y_test)

        errors.append(err)
        misClassifications.append(n_missed)


    # plt.plot(range(1, 100), errors)
    plt.plot(range(1, 100), misClassifications)

    plt.show()





if __name__ == '__main__':
    df = pd.read_csv('../Data/data_banknote_authentication.csv')
    random_forest_pt2(df)