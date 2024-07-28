import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report


def decision_tree_part1(df: pd.DataFrame):
    print(df.head())
    print(df["species"].unique())
    print(df.isnull().sum())
    print(df.info())
    df = df.dropna()
    print(df.info())
    print(df["sex"].unique())
    print(df[df["sex"] == "."])
    print(df[df["species"] == "Gentoo"].groupby("sex").describe().transpose())
    df.at[336, "sex"] = "FEMALE"
    print(df["sex"].unique())

    # sns.pairplot(df, hue='species')
    # sns.catplot(data=df, x='species', y='culmen_length_mm', kind='box')
    # plt.show()

    X = pd.get_dummies(df.drop("species", axis=1), drop_first=True)
    y = df["species"]

    decision_tree_part2(X, y)


def decision_tree_part2(X: pd.DataFrame, y: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101
    )

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    base_preds = model.predict(X_test)

    print(classification_report(y_test, base_preds))

    features = pd.DataFrame(
        index=X.columns,
        data=model.feature_importances_,
        columns=["Feature Importances"],
    ).sort_values("Feature Importances")
    print(features.head())

    report_model(model, X, X_test, y_test)

    pruned_tree = DecisionTreeClassifier(max_depth=2)
    pruned_tree.fit(X_train, y_train)

    report_model(pruned_tree, X, X_test, y_test)

    max_leaf_tree = DecisionTreeClassifier(max_leaf_nodes=3)
    max_leaf_tree.fit(X_train, y_train)

    report_model(max_leaf_tree, X, X_test, y_test)

    entropy_tree = DecisionTreeClassifier(criterion="entropy")
    entropy_tree.fit(X_train, y_train)
    report_model(entropy_tree, X, X_test, y_test)


def report_model(model, X, X_test, y_test):
    model_preds = model.predict(X_test)
    print(classification_report(y_test, model_preds))
    print("\n")
    plt.figure(figsize=(12, 8), dpi=200)
    plot_tree(model, feature_names=X.columns, filled=True, rounded=True)
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("../Data/penguins_size.csv")
    decision_tree_part1(df)
