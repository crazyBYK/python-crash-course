import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def random_forests_ex(df: pd.DataFrame):
    df = df.dropna()
    print(df.head())

    X = pd.get_dummies(df.drop('species', axis=1), drop_first=True)
    y = df['species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    rfc = RandomForestClassifier(n_estimators=10, max_features='sqrt', random_state=101)

    rfc.fit(X_train, y_train)
    pred = rfc.predict(X_test)
    print(pred)

    cfm = confusion_matrix(y_test, pred)

    print(cfm)

    cl_report = classification_report(y_test, pred)
    print(cl_report)






if __name__ == "__main__":
    df = pd.read_csv("../Data/penguins_size.csv")
    random_forests_ex(df)
