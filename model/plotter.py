from typing import Literal
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model.prophet_model import SalesModel
from model.tables_creator import TablesCreator
from utils.utils import sanitize_filename


class Plotter:
    def __init__(
        self,
        sales_model: SalesModel,
        tables_creator: TablesCreator,
        evaluation_period: Literal["month", "quarter", "year"] = "month",
    ) -> None:
        self.sales_model = sales_model
        self.tables_creator = tables_creator
        self.evaluation_period = evaluation_period
        self.forecast_table = self.create_forecast_table()
        self.summary_table = self.create_summary_table()
        self.perf_metrics_table = self.create_perf_metrics_table()
        self.summary_metrics_table = self.create_summary_metrics_table()
        self.figure = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Total Sales",
                "Filtered Sales",
                "Resampled Sales",
                "Metrics",
                "Sales Forecast",
                f"{self.evaluation_period}ly Forecast",
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
            specs=[
                [{"type": "scatter"}, {"type": "table"}],
                [{"type": "scatter"}, {"type": "table"}],
                [{"type": "scatter"}, {"type": "table"}],
            ],
            column_widths=[0.65, 0.35],
        )

    def create_summary_table(
        self,
    ):
        summary_table = go.Table(
            header=dict(
                values=self.tables_creator.formatted_summary_table.columns,
                fill_color="paleturquoise",
                align="left",
            ),
            cells=dict(
                values=[
                    self.tables_creator.formatted_summary_table[col]
                    for col in self.tables_creator.formatted_summary_table.columns
                ],
                fill_color="lavender",
                align="left",
            ),
        )
        return summary_table

    def create_forecast_table(
        self,
    ):
        forecast_table = go.Table(
            header=dict(
                values=[
                    self.evaluation_period,
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
                    self.tables_creator.formatted_forecast[
                        self.evaluation_period
                    ].astype(str),
                    self.tables_creator.formatted_forecast["y"],
                    self.tables_creator.formatted_forecast["yhat"],
                    self.tables_creator.formatted_forecast["yhat_lower"],
                    self.tables_creator.formatted_forecast["yhat_upper"],
                ],
                fill_color="lavender",
                align="left",
            ),
        )
        return forecast_table

    def create_perf_metrics_table(self):
        perf_metrics_table = go.Table(
            header=dict(
                values=self.sales_model.perf_metrics.columns,
                fill_color="paleturquoise",
                align="left",
            ),
            cells=dict(
                values=[
                    self.sales_model.perf_metrics[col]
                    for col in self.sales_model.perf_metrics.columns
                ],
                fill_color="lavender",
                align="left",
            ),
        )
        return perf_metrics_table

    def create_summary_metrics_table(self):
        # Create a table of evaluation metrics
        metrics_table = go.Table(
            header=dict(
                values=["Metric", "Value"], fill_color="paleturquoise", align="left"
            ),
            cells=dict(
                values=[
                    self.tables_creator.metrics_data["header"],
                    self.tables_creator.metrics_data["values"],
                ],
                fill_color="lavender",
                align="left",
            ),
        )
        return metrics_table

    def plot(self, forecast_plots_path: str):
        self.figure.add_trace(
            go.Scatter(
                x=self.sales_model.grouped_data["ds"],
                y=self.sales_model.resampled_data["y"],
                mode="lines",
                name="Unmodified Sales",
            ),
            row=1,
            col=1,
        )

        self.figure.add_trace(
            go.Scatter(
                x=self.sales_model.resampled_data["ds"],
                y=self.sales_model.resampled_data["y"],
                mode="lines",
                name="Resampled Sales",
            ),
            row=2,
            col=1,
        )

        # show actual sales
        self.figure.add_trace(
            go.Scatter(
                x=self.tables_creator.forecast_and_actuals["ds"],
                y=self.tables_creator.forecast_and_actuals["y"],
                mode="lines",
                name="Filtered Sales",
            ),
            row=3,
            col=1,
        )

        # show forecasted sales
        self.figure.add_trace(
            go.Scatter(
                x=self.tables_creator.forecast_and_actuals["ds"],
                y=self.tables_creator.forecast_and_actuals["yhat"],
                mode="lines",
                name="Forecast",
            ),
            row=3,
            col=1,
        )
        # show also the uncertainty in the forecast
        self.figure.add_trace(
            go.Scatter(
                x=self.tables_creator.forecast_and_actuals["ds"],
                y=self.tables_creator.forecast_and_actuals["yhat_lower"],
                fill=None,
                mode="lines",
                line_color="rgba(0,100,80,0.2)",
                name="Lower Bound",
            ),
            row=3,
            col=1,
        )
        self.figure.add_trace(
            go.Scatter(
                x=self.tables_creator.forecast_and_actuals["ds"],
                y=self.tables_creator.forecast_and_actuals["yhat_upper"],
                fill="tonexty",
                mode="lines",
                line_color="rgba(0,100,80,0.2)",
                name="Upper Bound",
            ),
            row=3,
            col=1,
        )

        # Add the tables to the plot
        self.figure.add_trace(self.summary_table, row=1, col=2)
        # self.figure.add_trace(self.perf_metrics_table, row=2, col=2)
        self.figure.add_trace(self.summary_metrics_table, row=2, col=2)
        self.figure.add_trace(self.forecast_table, row=3, col=2)

        self.figure.update_xaxes(
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

        self.figure.update_layout(
            title_text=f"Sales Forecasting - {" - ".join([v for v in self.sales_model.filters.values() if v is not None])}",
            showlegend=True,
        )

        self.figure.show()
        self.figure.write_html(
            os.path.join(
                forecast_plots_path,
                f"{sanitize_filename("_".join([v for v in self.sales_model.filters.values() if v is not None]))}_forecast.html",
            )
        )
