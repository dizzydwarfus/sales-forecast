# Built-in Imports
import sys
import os
from typing import Literal, List
import itertools

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

# Third-Party Imports
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
from prophet.serialize import model_to_json, model_from_json
from prophet.diagnostics import cross_validation, performance_metrics


class SalesModel(Prophet):
    def __init__(
        self,
        *args,
        sales_data: pd.DataFrame = None,
        region: Literal["Europe", "Americas", "APAC"] = None,
        business_line: Literal["AES", "PBF", "LFS", "AMS"] = None,
        product_family: str = None,
        product_subfamily: str = None,
        sales_channel: Literal[
            "DISTRIBUTOR & RESELLER", "MACHINE MAKER", "INDUSTRY", "SERVICE BUREAU"
        ] = None,
        target_col: Literal[
            "Total Sales", "Material Quantity (kg)", "Quantity in SKUs"
        ] = "Total Sales",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_col = target_col
        self.filters = {
            "Region": region,
            "BL Short": business_line,
            "Product Family": product_family,
            "Product Subfamily": product_subfamily,
            "Channel": sales_channel,
        }
        self._test_data = pd.read_csv(
            "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv",
            parse_dates=["ds"],
        )
        self.data = sales_data if sales_data is not None else self._test_data
        self.grouped_data = None
        self.resampled_data = None
        self.tunable_params = {
            "seasonality_mode": ["additive", "multiplicative"],
            "seasonality_prior_scale": [0.01, 10],
            "holidays_prior_scale": [0.01, 10],
            "changepoint_prior_scale": [0.001, 0.5],
            "changepoint_range": [0.8, 0.95],
        }
        self.all_params = [
            dict(zip(self.tunable_params.keys(), v))
            for v in itertools.product(*self.tunable_params.values())
        ]
        self.holidays = None
        self.future = None
        self.forecast_df = None
        self.cv_results = None
        self.perf_metrics = None
        self.tuning_df = None
        self._error = []

        self._post_init()

    # post-init to format and process sales data
    def _post_init(self):
        self.data = self._filter_sales_data(
            self._format_sales_data(self.data, self.target_col),
            self.filters,
        )
        self.grouped_data = self._group_sales_data(
            self.data,
            self.filters,
        )
        self.resampled_data = self._resample_sales_data(self.grouped_data, "D")

    @staticmethod
    def _format_sales_data(
        data: pd.DataFrame,
        date_col: str = "Date",
        target_col: Literal[
            "Total Sales", "Material Quantity (kg)", "Quantity in SKUs"
        ] = "Total Sales",
    ):
        if target_col is not None and date_col is not None:
            data = data.rename(columns={date_col: "ds", target_col: "y"})

        data["Year"] = data["ds"].dt.year
        data["Month"] = data["ds"].dt.month_name()
        data["Month"] = pd.Categorical(
            data["Month"],
            categories=[
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ],
            ordered=True,
        )
        return data

    @staticmethod
    def _filter_sales_data(
        raw_data: pd.DataFrame,
        filters: dict,
    ):
        raw_data = raw_data[raw_data["y"] >= 0]

        for k, v in filters.items():
            if v is not None:
                raw_data = raw_data.query(f"`{k}` == '{v}'")
        if raw_data.empty:
            raise ValueError("No data found for the given filters.")
        return raw_data

    @staticmethod
    def _group_sales_data(
        data: pd.DataFrame,
        filters: dict,
        agg: dict = {"y": "sum"},
    ):
        group_columns = ["ds"] + [k for k, v in list(filters.items()) if v is not None]
        grouped_data = data.groupby(group_columns).agg(agg).reset_index()
        return grouped_data

    @staticmethod
    def _resample_sales_data(
        data: pd.DataFrame,
        freq: Literal["D", "W", "M"] = "D",
    ):
        data = data.groupby("ds").sum().reset_index()
        data = data.resample(freq, on="ds").sum().reset_index()
        return data

    def create_holidays(
        self,
        year_list: list = None,
        country_list: List[str] = ["NL", "DE"],
    ):
        year_list = (
            year_list
            if year_list is not None
            else range(
                self.grouped_data["ds"].dt.year.min(),
                self.grouped_data["ds"].dt.year.max() + 3,
            )
        )
        holidays_dfs = []
        for country in country_list:
            holidays_dfs.append(make_holidays_df(year_list, country))

        self.holidays = pd.concat(holidays_dfs).drop_duplicates(subset=["ds"])

        return self.holidays

    def train(
        self,
    ):
        self.fit(self.resampled_data)

    def forecast(
        self,
        future_periods: int = 365,
        freq: Literal["D", "W", "M"] = "D",
        include_history: bool = True,
    ):
        self.future = self.make_future_dataframe(
            periods=future_periods, freq=freq, include_history=include_history
        )
        self.forecast_df = self.predict(self.future)

    def evaluate(
        self,
        initial: str = "730 days",  # two years worth of daily data
        period: str = "180 days",  # how much training data to increase for each validation
        horizon: str = "365 days",  # how far into the future to forecast
        **kwargs,
    ):
        self.cv_results = cross_validation(
            self,
            initial=initial,
            period=period,
            horizon=horizon,
            **kwargs,
        )
        self.perf_metrics = performance_metrics(self.cv_results)
        return self.perf_metrics

    def tune(
        self,
        initial: str = "730 days",
        period: str = "180 days",
        horizon: str = "365 days",
        error: Literal[
            "mse",
            "rmse",
            "mae",
            "mape",
            "mdape",
            "smape",
            "coverage",
        ] = "rmse",
        **kwargs,
    ) -> dict:
        self._error = []
        for params in self.all_params:
            self.model = self.__class__(**params)
            self.train(self.resampled_data)
            self.evaluate(initial=initial, period=period, horizon=horizon, **kwargs)
            self._error.append(self.perf_metrics[error].values[0])

        self.tuning_df = pd.DataFrame(self.all_params)
        self.tuning_df["rmse"] = self._rmse
        best_params = self.tuning_df[np.argmin(self._rmse)]

        # retrain model with best parameters from hyperparameter tuning
        self.model = self.__class__(**best_params)
        self.train(self.resampled_data)
        return best_params

    def save(self, filename: str):
        with open(filename, "w") as fout:
            fout.write(model_to_json(self))  # Save model

    @classmethod
    def load(cls, filename: str):
        with open(filename, "r") as fin:
            loaded_model = model_from_json(fin.read())  # Load model

        sales_model = cls()
        sales_model.__dict__.update(loaded_model.__dict__)
        return sales_model
