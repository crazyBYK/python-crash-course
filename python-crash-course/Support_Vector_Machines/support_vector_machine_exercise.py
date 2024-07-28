import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def exercise_pt1(df: pd.DataFrame):
    print(df.head())
    # TASK : What are the unique variables in the target column we are trying to predict(quality)?
    print(df["quality"].unique())

    #     TASK: Creat a countplot that displays the count per category of Legit vs Fraud.
    #           Is the label/target balanced or unbalanced?

    # sns.countplot(df, x='quality')
    # plt.show()

    #   TASK:
    #     sns.countplot(df, x='type', hue='quality')
    # plt.show()

    #  TASK: What percentage of red wines are Fraud? What percentage of white wines are fraud?
    reds = df[df["type"] == "red"]
    # red_percentage = (reds['quality'].value_counts('Fraud')/len(reds)) * 100
    red_percentage = (len(reds[reds["quality"] == "Fraud"]) / len(reds)) * 100
    # print(f'Percentage of fraud in Red Wines: {red_percentage}')
    # print(red)
    # red_percentage = (df['type'].value_counts('red')/len(df)) * 100
    # print('Percentage of red ')
    whites = df[df["type"] == "white"]
    white_percentage = (len(whites[whites["quality"] == "Fraud"]) / len(whites)) * 100
    # print(f'Percentage of fraud in White Wines: {white_percentage}')

    #   TASK: Calculate the correlations between the various feature and the 'quality' column.
    #   To do this may need the column to 0 and 1 instead of a String.
    df["Fraud"] = df["quality"].map({"Fraud": 1, "Legit": 0})
    #     df.corr(numeric_only=True)['Fraud'][:-1].sort_values().plot(kind='bar')
    #     plt.ylim([-0.1,0.15])
    #     plt.show()

    # test_data = df.corr(numeric_only=True)['Fraud'][:-1].sort_values()
    #
    # print(test_data)

    #     TASK: Create a clustermap with seaborn to explore the relationships between variables.
    sns.clustermap(df.corr(numeric_only=True))
    plt.show()


def exercise_pt2(df: pd.DataFrame):
    df["Fraud"] = df["quality"].map({"Fraud": 1, "Legit": 0})
    df["type"] = pd.get_dummies(df["type"], drop_first=True)

    # TASK: Separate out the data into X features and y target label('quality' column)
    df = df.drop("Fraud", axis=1)
    X = df.drop("quality", axis=1)
    y = df["quality"]

    # TASK: Perform a Train|Test split on the data, with a 10% test size and 101 random state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=101
    )

    # TASK: Scale the X train and X test data.
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_x_test = scaler.transform(X_test)

    svc = SVC(class_weight="balanced")
    param_grid = {"C": [0.001, 0.01, 0.1, 0.5, 1], "gamma": ["scale", "auto"]}

    grid = GridSearchCV(svc, param_grid)
    grid.fit(scaled_X_train, y_train)
    print(grid.best_params_)

    grid_predict = grid.predict(scaled_x_test)

    ac_score = accuracy_score(y_test, grid_predict)
    cf_matrix = confusion_matrix(y_test, grid_predict)
    cl_report = classification_report(y_test, grid_predict)

    print(f"accuracy score: {ac_score}")
    print(f"confusion matrix: {cf_matrix}")
    print(f"classificaiton_report: {cl_report}")


if __name__ == "__main__":
    df = pd.read_csv("../Data/wine_fraud.csv")
    # exercise_pt1(df)
    exercise_pt2(df)
