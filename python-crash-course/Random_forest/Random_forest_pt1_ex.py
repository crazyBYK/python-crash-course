import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Using Machine_learning for feature analysing
def random_forest_pt1_ex(df: pd.DataFrame):
    print(df.head())
    # sns.countplot(data=df, x='class')
    # plt.show()

    print(df.describe())

    feat_uni = df.describe().T.reset_index().sort_values('unique')
    print(feat_uni)
    # sns.barplot(data=feat_uni, x='index', y='unique')
    # plt.xticks(rotation=90)
    # plt.show()

    X = df.drop('class', axis=1)
    X = pd.get_dummies(X, drop_first=True)

    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)

    model = AdaBoostClassifier(n_estimators=1)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    print(classification_report(y_test, prediction))
    print(model.feature_importances_)
    print(model.feature_importances_.argmax())
    print(X.columns[22])

    # sns.countplot(data=df, x='odor', hue='class')
    # plt.show()

    error_rates = []

    for n in range(1, 96):
        model = AdaBoostClassifier(n_estimators=n)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        error = 1 - accuracy_score(y_test, preds)

        error_rates.append(error)
    print(error_rates)

    plt.plot(range(1, 96), error_rates)
    plt.show()

    feat = pd.DataFrame(index=X.columns, data = model.feature_importances_, columns=['Importance'])
    imp_feats = feat[feat['Importance'] > 0]
    sns.barplot(data=imp_feats.sort_values('Importance'), x=imp_feats.index, y='Importance')
    plt.xticks(rotation=90)


    pass



if __name__ == '__main__':
    df = pd.read_csv('../Data/mushrooms.csv')
    random_forest_pt1_ex(df)