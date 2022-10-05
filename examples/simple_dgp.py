import os

import numpy as np
import pandas as pd
from anatomy import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def train_ols(x_train: pd.DataFrame, y_train: pd.Series) -> AnatomyModel:
    ols_model = LinearRegression().fit(x_train, y_train)

    def pred_fn_ols(xs: np.ndarray) -> np.ndarray:
        xs_df = pd.DataFrame(xs, columns=x_train.columns)
        return np.array(ols_model.predict(xs_df)).flatten()

    return AnatomyModel(pred_fn_ols)


def train_rf(x_train: pd.DataFrame, y_train: pd.Series) -> AnatomyModel:
    rf_model = RandomForestRegressor(random_state=1338).fit(x_train, y_train)

    def pred_fn_rf(xs: np.ndarray) -> np.ndarray:
        xs_df = pd.DataFrame(xs, columns=x_train.columns)
        return np.array(rf_model.predict(xs_df)).flatten()

    return AnatomyModel(pred_fn_rf)


def generate_data() -> [pd.DataFrame, AnatomySubsets]:

    xy = pd.DataFrame(np.random.normal(0, 1, (500, 3)), columns=["x_0", "x_1", "x_2"])
    xy["y"] = xy.sum(axis=1) + np.random.normal(0, 1, 500)

    # set a unique and monotonically increasing index (default index would suffice):
    xy.index = pd.date_range("2021-04-19", "2022-08-31").map(lambda x: x.date())

    subsets = AnatomySubsets.generate(
        index=xy.index,
        initial_window=100,
        estimation_type=AnatomySubsets.EstimationType.EXPANDING,
        periods=1,
        gap=0
    )

    return xy, subsets


def estimate_anatomy(xy: pd.DataFrame, subsets: AnatomySubsets) -> None:

    def mapper(key: AnatomyModelProvider.PeriodKey) -> \
            AnatomyModelProvider.PeriodValue:
        train = xy.iloc[subsets.get_train_subset(key.period)]
        test = xy.iloc[subsets.get_test_subset(key.period)]

        if key.model_name == "ols":
            model = train_ols(train.drop("y", axis=1), train["y"])
        elif key.model_name == "rf":
            model = train_rf(train.drop("y", axis=1), train["y"])

        return AnatomyModelProvider.PeriodValue(train, test, model)

    provider = AnatomyModelProvider(
        n_periods=subsets.n_periods,
        n_features=xy.shape[1] - 1,
        model_names=["ols", "rf"],
        y_name="y",
        provider_fn=mapper
    )

    Anatomy(provider=provider, n_iterations=10).precompute(
        n_jobs=max(1, min(os.cpu_count(), 16)), save_path="anatomy.bin"
    )


def anatomize(xy: pd.DataFrame, subsets: AnatomySubsets) -> None:

    anatomy = Anatomy.load("anatomy.bin")

    groups = {
        "rf": ["rf"],
        "ols": ["ols"],
        "ols+rf": ["ols", "rf"]
    }

    def anatomize_r2_oos() -> pd.DataFrame:

        prevailing_mean = np.array([
            xy.iloc[subsets.get_train_subset(period=i)]["y"].mean()
            for i in range(subsets.n_periods)
        ])

        def transform(y_hat, y):
            return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - prevailing_mean) ** 2)

        return anatomy.explain(
            model_sets=AnatomyModelCombination(groups=groups),
            transformer=AnatomyModelOutputTransformer(transform=transform)
        )

    def anatomize_rmse() -> pd.DataFrame:

        def transform(y_hat, y):
            return np.sqrt(np.mean((y - y_hat) ** 2))

        return anatomy.explain(
            model_sets=AnatomyModelCombination(groups=groups),
            transformer=AnatomyModelOutputTransformer(transform=transform)
        )

    def anatomize_rmse_subperiod() -> pd.DataFrame:

        def transform(y_hat, y):
            return np.sqrt(np.mean((y - y_hat) ** 2))

        return anatomy.explain(
            model_sets=AnatomyModelCombination(groups=groups),
            transformer=AnatomyModelOutputTransformer(transform=transform),
            explanation_subset=pd.date_range("2021-07-28", "2021-08-06").map(lambda x: x.date())
        )

    def anatomize_mae() -> pd.DataFrame:

        def transform(y_hat, y):
            return np.mean(np.abs(y - y_hat))

        return anatomy.explain(
            model_sets=AnatomyModelCombination(groups=groups),
            transformer=AnatomyModelOutputTransformer(transform=transform)
        )

    def anatomize_se() -> pd.DataFrame:

        def transform(y_hat, y):
            return (y - y_hat) ** 2

        return anatomy.explain(
            model_sets=AnatomyModelCombination(groups=groups),
            transformer=AnatomyModelOutputTransformer(transform=transform)
        )

    def anatomize_forecasts() -> pd.DataFrame:

        def transform(y_hat):
            return y_hat

        return anatomy.explain(
            model_sets=AnatomyModelCombination(groups=groups),
            transformer=AnatomyModelOutputTransformer(transform=transform)
        )

    pd.options.display.width = 0  # to print all columns

    tmpl = "contributions to %s:"

    print(tmpl % "out-of-sample R²")
    print(anatomize_r2_oos(), "\n")

    print(tmpl % "root mean squared error")
    print(anatomize_rmse(), "\n")

    print(tmpl % "root mean squared error (subperiod)")
    print(anatomize_rmse_subperiod(), "\n")

    print(tmpl % "mean absolute error")
    print(anatomize_mae(), "\n")

    print(tmpl % "squared errors")
    print(anatomize_se(), "\n")

    print(tmpl % "forecasts")
    print(anatomize_forecasts(), "\n")


def main():

    # set random seed for reproducibility:
    np.random.seed(1338)

    # first generate data from the simple linear dgp:
    xy, subsets = generate_data()

    # next estimate the models and the anatomy, which is as "anatomy.bin" in the current working directory:
    estimate_anatomy(xy, subsets)

    # finally load "anatomy.bin" and anatomize the estimated models on a multitude of metrics:
    anatomize(xy, subsets)


if __name__ == "__main__":
    main()
