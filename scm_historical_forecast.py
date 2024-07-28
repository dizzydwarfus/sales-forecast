# Built-in Imports
import sys
import os
from typing import Literal
from dotenv import load_dotenv

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

# Third-Party Imports
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Internal Imports
from utils._constants import (
    PAYLOAD,
    DOMAIN,
    API_VERSION,
)
from utils.utils import get_data, format_euro

load_dotenv(".env")

from utils.utils import (
    format_euro,
    generate_auth_header,
    get_sales_data,
)


def model_and_plot(
    sales_data: pd.DataFrame,
    filters: dict = {},
    x: Literal["Year", "Month", "Date"] = "Date",
    y: Literal[
        "Total Sales", "Material Quantity (kg)", "Quantity in SKUs"
    ] = "Total Sales",
    dimension: str = "Dimension",
    resampling_period: str = "D",  # should be same as frequency
    future_period: int = 365,  # unit based on frequency
    frequency: str = "D",  # should be same as resampling_period
    prophet_kwargs: dict = {},
    sales_grouping_period: Literal["month", "quarter", "year"] = "month",
) -> pd.DataFrame:
    sales_data["Year"] = sales_data["Date"].dt.year
    sales_data["Month"] = sales_data["Date"].dt.month_name()
    sales_data["Month"] = pd.Categorical(
        sales_data["Month"],
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

    # Remove negative sales values
    sales_data = sales_data[sales_data[y] >= 0]

    # filter out data based on kwargs
    if filters != {}:
        filtered_sales_data = sales_data.query(
            " & ".join([f"`{key}` == '{value}'" for key, value in filters.items()])
        )
        if filtered_sales_data.empty:
            raise ValueError("No data found for the given filters.")

    # create group columns list
    group_columns = [x] + list(filters.keys())
    agg_sales_data = filtered_sales_data.groupby(group_columns)[y].sum().reset_index()

    # Calculate additional summary metrics
    total_sales = filtered_sales_data[y].sum()
    average_sales = filtered_sales_data[y].mean()
    sales_std_dev = filtered_sales_data[y].std()
    number_of_days_ordered = filtered_sales_data["Date"].nunique()
    order_frequency = len(filtered_sales_data) / number_of_days_ordered
    median_sales = filtered_sales_data[y].median()
    total_transactions = len(filtered_sales_data)
    sales_variance = filtered_sales_data[y].var()
    average_order_value = total_sales / total_transactions
    sales_skewness = filtered_sales_data[y].skew()
    sales_kurtosis = filtered_sales_data[y].kurt()
    unique_customers = filtered_sales_data["Account Id"].nunique()
    repeat_customers = (
        filtered_sales_data.groupby("Account Id")
        .filter(lambda x: len(x) > 1)["Account Id"]
        .nunique()
    )
    repeat_customer_rate = (
        repeat_customers / unique_customers if unique_customers > 0 else 0
    )

    # Create a summary of sales data in filtered_sales_data
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
    filtered_sales_summary = pd.DataFrame(summary_data)

    # Create summary table for the plot
    filtered_sales_table = go.Table(
        header=dict(
            values=filtered_sales_summary.columns,
            fill_color="paleturquoise",
            align="left",
        ),
        cells=dict(
            values=[
                filtered_sales_summary[col] for col in filtered_sales_summary.columns
            ],
            fill_color="lavender",
            align="left",
        ),
    )

    if not agg_sales_data.empty:
        agg_sales_data["Dimension"] = (
            agg_sales_data[list(filters.keys())].astype(str).agg(" - ".join, axis=1)
        )

    resampled_df = (
        agg_sales_data.groupby(["Date"])["Total Sales"]
        .sum()
        .reset_index()
        .resample(resampling_period, on="Date")
        .sum()
        .reset_index()
    )

    resampled_df = resampled_df.rename(columns={"Date": "ds", "Total Sales": "y"})

    # fit linear regression model
    model_prophet = Prophet(**prophet_kwargs)

    model_prophet.fit(resampled_df)

    df_future = model_prophet.make_future_dataframe(
        periods=future_period, freq=frequency
    )

    forecast_prophet = model_prophet.predict(df_future)

    # combine test and forecast data
    forecast_prophet = forecast_prophet.merge(
        resampled_df[["ds", "y"]],
        on="ds",
        how="inner",
        suffixes=("_forecast", "_actual"),
    )

    # Calculate evaluation metrics
    total_forecasted_sales = forecast_prophet["yhat"].sum()
    y_true = forecast_prophet["y"].values
    y_pred = forecast_prophet["yhat"].values
    mae = mean_absolute_error(y_true[: len(y_pred)], y_pred)
    mape = (
        (forecast_prophet["yhat"] - forecast_prophet["y"]).abs() / forecast_prophet["y"]
    ).mean()
    mape = np.mean(
        np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))
    )  # Adjusting for zero values
    rmse = np.sqrt(mean_squared_error(y_true[: len(y_pred)], y_pred))

    # Create a table of evaluation metrics
    metrics_table = go.Table(
        header=dict(
            values=["Metric", "Value"], fill_color="paleturquoise", align="left"
        ),
        cells=dict(
            values=[
                ["MAE", "RMSE", "MAPE", "Total Sales", "Total Forecasted Sales"],
                [
                    format_euro(mae),
                    format_euro(rmse),
                    f"{mape:.2%}",
                    format_euro(total_sales),
                    format_euro(total_forecasted_sales),
                ],
            ],
            fill_color="lavender",
            align="left",
        ),
    )

    # Aggregate the forecasted sales by period
    forecast_prophet["month"] = forecast_prophet["ds"].dt.to_period("M")
    forecast_prophet["quarter"] = forecast_prophet["ds"].dt.to_period("Q")
    forecast_prophet["year"] = forecast_prophet["ds"].dt.to_period("Y")

    def create_forecast_table(df: pd.DataFrame, period: str) -> pd.DataFrame:
        forecast_table = (
            df.groupby(period)
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

    period_forecast, formatted_period_forecast = create_forecast_table(
        forecast_prophet, sales_grouping_period
    )
    # Create a table for sales_grouping_period forecast
    forecast_table = go.Table(
        header=dict(
            values=[
                sales_grouping_period,
                "Actual Sales",
                "Forecasted Sales",
                "Lower Bound",
                "Upper Bound",
            ],
            fill_color="paleturquoise",
            align="left",
        ),
        cells=dict(
            values=[
                formatted_period_forecast[sales_grouping_period].astype(str),
                formatted_period_forecast["y"],
                formatted_period_forecast["yhat"],
                formatted_period_forecast["yhat_lower"],
                formatted_period_forecast["yhat_upper"],
            ],
            fill_color="lavender",
            align="left",
        ),
    )

    # create plotly figure with subplots
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Total Sales",
            "Filtered Sales",
            "Resampled Sales",
            "Metrics",
            "Sales Forecast",
            f"{sales_grouping_period}ly Forecast",
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
        specs=[
            [{"type": "scatter"}, {"type": "table"}],
            [{"type": "scatter"}, {"type": "table"}],
            [{"type": "scatter"}, {"type": "table"}],
        ],
        column_widths=[0.75, 0.25],
    )

    # plot resampled sales data on a different plot
    for dim in agg_sales_data[dimension].unique():
        fig.add_trace(
            go.Scatter(
                x=agg_sales_data[agg_sales_data[dimension] == dim][x],
                y=agg_sales_data[agg_sales_data[dimension] == dim][y],
                mode="lines",
                name="Unmodified Sales",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=resampled_df["ds"],
            y=resampled_df["y"],
            mode="lines",
            name="Resampled Sales",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_prophet["ds"],
            y=forecast_prophet["y"],
            mode="lines",
            name="Actual",
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_prophet["ds"],
            y=forecast_prophet["yhat"],
            mode="lines",
            name="Forecast",
        ),
        row=3,
        col=1,
    )

    # show also the uncertainty in the forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_prophet["ds"],
            y=forecast_prophet["yhat_lower"],
            fill=None,
            mode="lines",
            line_color="rgba(0,100,80,0.2)",
            name="Lower Bound",
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_prophet["ds"],
            y=forecast_prophet["yhat_upper"],
            fill="tonexty",
            mode="lines",
            line_color="rgba(0,100,80,0.2)",
            name="Upper Bound",
        ),
        row=3,
        col=1,
    )

    # Add the tables to the plot
    fig.add_trace(filtered_sales_table, row=1, col=2)
    fig.add_trace(metrics_table, row=2, col=2)
    fig.add_trace(forecast_table, row=3, col=2)

    fig.update_xaxes(
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=2, label="2y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
    )

    fig.update_layout(
        title=f"Sales Forecast Evaluation - {" - ".join(filters.values())}",
        xaxis_title=x,
        yaxis_title=y,
        showlegend=True,
    )

    fig.show()

    return (
        filtered_sales_data,
        agg_sales_data,
        resampled_df,
        forecast_prophet,
        period_forecast,
        filtered_sales_summary,
    )


if __name__ == "__main__":
    FOLDER_PATH = os.getenv("SNOP_BREAKDOWN_PATH")
    sales_path = os.path.join(
        FOLDER_PATH, "data", "SCM Forecast - Sales Topic_Summary.csv"
    )
    auth_header = generate_auth_header(DOMAIN, PAYLOAD)
    api_endpoint = DOMAIN + f"/services/data/v{API_VERSION}/query/?q="
    # forecast_data = get_forecast_data(DOMAIN, api_endpoint, auth_header)

    sales_data = get_sales_data(sales_path)
    filters = {
        "Region": "Europe",
        "BL Short": "LFS",
        "Product Family": "Tough",
        # "Product Subfamily": "ST 45",
        # "Channel": "Industry",
    }
    x = "Date"
    y = "Total Sales"

    DE_holiday_df = make_holidays_df(
        year_list=[2019 + i for i in range(10)], country="DE"
    )
    NL_holiday_df = make_holidays_df(
        year_list=[2019 + i for i in range(10)], country="NL"
    )

    holiday_df = (
        pd.concat([DE_holiday_df, NL_holiday_df])
        .sort_values(by="ds")
        .drop_duplicates(subset="ds", keep="first")
    )
    prophet_kwargs = {
        "growth": "linear",
        "changepoint_prior_scale": 1.0,
        "seasonality_mode": "multiplicative",
        "holidays": holiday_df,
    }

    (
        filtered_sales_data,
        agg_sales_data,
        resampled_df,
        prophet_df,
        period_forecast,
        filtered_sales_summary,
    ) = model_and_plot(
        sales_data=sales_data,
        x=x,
        y=y,
        filters=filters,
        dimension="Dimension",
        resampling_period="1D",
        future_period=1,
        frequency="1D",
        prophet_kwargs=prophet_kwargs,
        sales_grouping_period="quarter",
    )

    with pd.ExcelWriter(
        os.path.join(FOLDER_PATH, "forecasted_sales_data.xlsx")
    ) as writer:
        filtered_sales_data.to_excel(
            writer, sheet_name="Filtered Sales Data", index=False
        )
        agg_sales_data.to_excel(writer, sheet_name="Aggregated Sales Data", index=False)
        resampled_df.to_excel(writer, sheet_name="Resampled Sales Data", index=False)
        prophet_df.to_excel(writer, sheet_name="Predicted Sales Data", index=False)
        period_forecast.to_excel(writer, sheet_name="Monthly Forecast", index=False)
        filtered_sales_summary.to_excel(writer, sheet_name="Summary", index=False)

# TODO: Hyparamter tune for each dimension combination
# TODO: Use Prophet's built-in cross validation for each dimension combination
# TODO: Aggregate all results into a single dataframe and save to excel
# TODO: Create streamlit app for interactive forecasting and evaluation - input: sales_data, filters, parameters output: plot, summary, evaluation
