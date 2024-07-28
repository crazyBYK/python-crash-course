import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def random_forest_pt3(df: pd.DataFrame):
    print(df.head())
    df.columns = ['Signal', 'Density']
    sns.scatterplot(x='Signal', y='Density', data=df)
    # plt.show()
    X = df['Signal'].values.reshape(-1, 1)
    y = df['Density']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

    lr_model = LinearRegression()

    run_model(df, lr_model, X_train, X_test, y_train, y_test)

def random_forest_polynomial_regression(df: pd.DataFrame):
    df.columns = ['Signal', 'Density']
    X = df['Signal'].values.reshape(-1, 1)
    y = df['Density']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

    pip = make_pipeline(PolynomialFeatures(degree=6), LinearRegression())

    run_model(df, pip, X_train, X_test, y_train, y_test)





def run_model(df, model, X_train, X_test, y_train, y_test):

#     FIT MODEL TRAINING
    model.fit(X_train, y_train)
#     GET METRICS
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')
#     PLOT RESULT MODEL SIGNAL RANGE
    signal_range = np.arange(0, 100)
    signal_pred = model.predict(signal_range.reshape(-1, 1))

    plt.figure(figsize=(12, 8), dpi=200)
    plt.scatter(x='Signal', y='Density', data=df, color='black')
    plt.plot(signal_range, signal_pred)
    plt.show()


    





if __name__ == '__main__':
    df = pd.read_csv('../Data/rock_density_xray.csv')
    # random_forest_pt3(df)
    random_forest_polynomial_regression(df)