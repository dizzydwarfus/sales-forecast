import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from utils.utils import format_euro

from typing import Literal


class TablesCreator:
    def __init__(
        self,
        filtered_sales_data: pd.DataFrame,
        grouped_sales_data: pd.DataFrame,
        resampled_sales_data: pd.DataFrame,
        forecasted_sales_data: pd.DataFrame,
        evaluation_period: Literal["month", "quarter", "year"] = "month",
    ) -> None:
        self.sales_data = filtered_sales_data
        self.grouped_sales_data = grouped_sales_data
        self.resampled_sales_data = resampled_sales_data
        self.forecasted_sales_data = forecasted_sales_data
        self.forecast_and_actuals = self.forecasted_sales_data.merge(
            self.resampled_sales_data[["ds", "y"]],
            on="ds",
            how="left",
            suffixes=("_forecast", "_actual"),
        )
        # TODO: replace all nan with 0
        self.summary_table, self.formatted_summary_table = self._create_summary_table()
        self.forecast_table, self.formatted_forecast = self._create_forecast_table(
            period=evaluation_period
        )
        self.metrics_data = self._create_metrics()

    def _create_summary_table(self) -> pd.DataFrame:
        # Calculate additional summary metrics
        total_sales = self.sales_data["y"].sum()
        average_sales = self.sales_data["y"].mean()
        sales_std_dev = self.sales_data["y"].std()
        number_of_days_ordered = self.sales_data["ds"].nunique()
        order_frequency = len(self.sales_data) / number_of_days_ordered
        median_sales = self.sales_data["y"].median()
        total_transactions = len(self.sales_data)
        sales_variance = self.sales_data["y"].var()
        average_order_value = total_sales / total_transactions
        sales_skewness = self.sales_data["y"].skew()
        sales_kurtosis = self.sales_data["y"].kurt()
        unique_customers = self.sales_data["Account Id"].nunique()
        repeat_customers = (
            self.sales_data.groupby("Account Id")
            .filter(lambda x: len(x) > 1)["Account Id"]
            .nunique()
        )
        repeat_customer_rate = (
            repeat_customers / unique_customers if unique_customers > 0 else 0
        )

        # Create a summary of sales data in self.sales_data
        summary_data = {
            "Metric": [
                "Total Sales",
                "Average Sales",
                "Sales Std Dev",
                "Number of Days Ordered",
                "Order Frequency",
                "Median Sales",
                "Total Transactions",
                "Sales Variance",
                "Average Order Value",
                "Sales Skewness",
                "Sales Kurtosis",
                "Unique Customers",
                "Repeat Customer Rate",
            ],
            "Value": [
                round(total_sales),
                round(average_sales),
                round(sales_std_dev),
                number_of_days_ordered,
                round(order_frequency),
                median_sales,
                total_transactions,
                round(sales_variance),
                round(average_order_value),
                round(sales_skewness),
                round(sales_kurtosis),
                unique_customers,
                round(repeat_customer_rate),
            ],
            "Unit": [
                "€",
                "€",
                "€",
                "days",
                "orders/day",
                "€",
                "transactions",
                "€",
                "€",
                "skewness",
                "kurtosis",
                "customers",
                "%",
            ],
        }
        summary_df = pd.DataFrame(summary_data)

        formatted_summary = {
            "Metric": [
                "Total Sales",
                "Average Sales",
                "Sales Std Dev",
                "Number of Days Ordered",
                "Order Frequency",
                "Median Sales",
                "Total Transactions",
                "Sales Variance",
                "Average Order Value",
                "Sales Skewness",
                "Sales Kurtosis",
                "Unique Customers",
                "Repeat Customer Rate",
            ],
            "Value": [
                format_euro(total_sales),
                format_euro(average_sales),
                format_euro(sales_std_dev),
                f"{number_of_days_ordered:,.0f}",
                round(order_frequency),
                format_euro(median_sales),
                f"{total_transactions:,.0f}",
                format_euro(sales_variance),
                format_euro(average_order_value),
                round(sales_skewness),
                round(sales_kurtosis),
                f"{unique_customers:,.0f}",
                f"{repeat_customer_rate:.0%}",
            ],
            "Unit": [
                "€",
                "€",
                "€",
                "days",
                "orders/day",
                "€",
                "transactions",
                "€",
                "€",
                "skewness",
                "kurtosis",
                "customers",
                "%",
            ],
        }

        formatted_summary_df = pd.DataFrame(formatted_summary)

        return summary_df, formatted_summary_df

    def _create_forecast_table(self, period: str) -> pd.DataFrame:
        self.forecast_and_actuals["month"] = self.forecast_and_actuals[
            "ds"
        ].dt.to_period("M")
        self.forecast_and_actuals["quarter"] = self.forecast_and_actuals[
            "ds"
        ].dt.to_period("Q")
        self.forecast_and_actuals["year"] = self.forecast_and_actuals[
            "ds"
        ].dt.to_period("Y")

        forecast_table = (
            self.forecast_and_actuals.groupby(period)
            .agg(
                {
                    "y": "sum",
                    "yhat": "sum",
                    "yhat_lower": "sum",
                    "yhat_upper": "sum",
                }
            )
            .reset_index()
        )
        formatted_forecast = forecast_table.copy()

        formatted_forecast["y"] = formatted_forecast["y"].apply(format_euro)
        formatted_forecast["yhat"] = formatted_forecast["yhat"].apply(format_euro)
        formatted_forecast["yhat_lower"] = formatted_forecast["yhat_lower"].apply(
            format_euro
        )
        formatted_forecast["yhat_upper"] = formatted_forecast["yhat_upper"].apply(
            format_euro
        )
        return forecast_table, formatted_forecast

    def _create_metrics(self) -> pd.DataFrame:
        # Calculate evaluation metrics
        total_sales = self.forecast_and_actuals["y"].sum()
        total_forecasted_sales = self.forecast_and_actuals["yhat"].sum()
        y_true = self.forecast_and_actuals["y"].values
        y_pred = self.forecast_and_actuals["yhat"].values
        mae = mean_absolute_error(y_true[: len(y_pred)], y_pred)
        mape = (
            (self.forecast_and_actuals["yhat"] - self.forecast_and_actuals["y"]).abs()
            / self.forecast_and_actuals["y"]
        ).mean()
        mape = np.mean(
            np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))
        )  # Adjusting for zero values
        rmse = np.sqrt(mean_squared_error(y_true[: len(y_pred)], y_pred))

        # output_dict
        metrics_data = {
            "header": ["MAE", "RMSE", "MAPE", "Total Sales", "Total Forecasted Sales"],
            "values": [
                format_euro(mae),
                format_euro(rmse),
                f"{mape:.2%}",
                format_euro(total_sales),
                format_euro(total_forecasted_sales),
            ],
        }
        return metrics_data
