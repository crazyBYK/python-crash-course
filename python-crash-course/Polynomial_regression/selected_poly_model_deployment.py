import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from joblib import dump, load


df = pd.read_csv("Advertising.csv")


def deploy_selected_poly_model():
    X = df.drop("sales", axis=1)
    y = df["sales"]

    final_converter = PolynomialFeatures(degree=3, include_bias=False)

    final_model = LinearRegression()

    final_conveted_X = final_converter.fit_transform(X)
    final_model.fit(final_conveted_X, y)

    dump(final_model, "final_poly_model.joblib")
    dump(final_converter, "final_poly_converter.joblib")


def load_converter_model():
    loaded_converter = load("final_poly_converter.joblib")
    loaded_model = load("final_poly_model.joblib")

    campaign = [[149, 22, 12]]

    # print(loaded_converter.fit_transform(campaign).shape)
    transformed_data = loaded_converter.fit_transform(campaign)
    campaign_predicted = loaded_model.predict(transformed_data)
    print(campaign_predicted)


if __name__ == "__main__":
    # deploy_selected_poly_model()
    load_converter_model()
