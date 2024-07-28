import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def Boosting_ex(df: pd.DataFrame):
    print(df.head())
    # sns.countplot(data=df, x='class')
    # plt.show()
    print(df.describe())
    print(df.describe().transpose().reset_index().sort_values('unique'))
    feat_uni = df.describe().transpose().reset_index().sort_values('unique')
    plt.figure(figsize=(14, 6), dpi=200)
    sns.barplot(data=feat_uni, x='index', y='unique')
    plt.xticks(rotation=90)
    # plt.show()

    X = df.drop('class', axis=1)
    X = pd.get_dummies(X, drop_first=True)
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)

    model = AdaBoostClassifier(n_estimators=1)
    model.fit(X_train, y_train)
    mode_preds = model.predict(X_test)
    cl = classification_report(y_test, mode_preds)
    print(model.feature_importances_)
    print(cl)




if __name__ == '__main__':
    df = pd.read_csv('../Data/mushrooms.csv')
    Boosting_ex(df)