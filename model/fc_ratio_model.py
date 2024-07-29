# Third-Party Imports
import pandas as pd

# Internal Imports
from utils.utils import get_forecast_data


class ForecastRatioModel:
    def __init__(
        self,
        sales_data: pd.DataFrame,
        domain: str,
        auth_header: dict,
        api_endpoint: str,
        forecast_query: str,
    ):
        self.domain = domain
        self.auth_header = auth_header
        self.api_endpoint = api_endpoint
        self.forecast_query = forecast_query
        self.actual_data = sales_data
        self.forecast_data = self._get_forecast_data()
        self.forecast_sku = None
        self.forecast_sku_actual_sales = None

    def _get_forecast_data(
        self,
    ):
        self.forecast_data = get_forecast_data(
            domain=self.domain,
            api_endpoint=self.api_endpoint,
            auth_header=self.auth_header,
        )
        return self.forecast_data

    def forecast(
        self,
        train_start: str = "2020-01-01",
        train_end: str = "2023-06-01",
        forecast_start: str = "2023-06-01",
        forecast_end: str = "2024-07-01",
    ):
        train_start, train_end, forecast_start, forecast_end = [
            train_start,
            train_end,
            forecast_start,
            forecast_end,
        ]

        # use past x years data
        historical_sales = self.actual_data.query(
            "Date >= @train_start & Date < @train_end"
        ).copy()

        # use forecast data for the next year (this determines the forecast period)
        forecast = self.forecast_data.query(
            "Date__c >= @forecast_start & Date__c < @forecast_end"
        ).copy()

        # Extract month and year for grouping
        historical_sales["Month"] = historical_sales["Date"].dt.month_name()
        forecast["Month"] = forecast["Date__c"].dt.month_name()

        # Calculate total sales per SKU per grouping
        sku_sales_totals = (
            historical_sales.groupby(
                [
                    "Account Id",
                    "Account Name",
                    "BL Short",
                    "Product Family",
                    "Month",
                    "Product Id",
                    "Local Item Code",
                    "Local Item Description",
                ]
            )["Total Sales"]
            .sum()
            .reset_index()
        )

        group_sales_totals = (
            historical_sales.groupby(
                ["Account Id", "Account Name", "BL Short", "Product Family", "Month"]
            )["Total Sales"]
            .sum()
            .reset_index()
        )

        # Merge to calculate SKU percentage
        sku_sales_percentage = pd.merge(
            sku_sales_totals,
            group_sales_totals,
            on=["Account Id", "BL Short", "Product Family", "Month"],
            suffixes=("_sku", "_total"),
        )
        sku_sales_percentage["SKU_Percentage"] = (
            sku_sales_percentage["Total Sales_sku"]
            / sku_sales_percentage["Total Sales_total"]
        )

        # Merge forecast data with SKU percentage distribution
        forecast = forecast.rename(
            columns={
                "Account__c": "Account Id",
                "Business_line__c": "BL Short",
                "Product_Family__c": "Product Family",
            }
        )
        self.forecast_sku = pd.merge(
            forecast,
            sku_sales_percentage,
            on=[
                "Account Id",
                "BL Short",
                "Product Family",
                "Month",
            ],
            how="left",
        )

        # Allocate forecast amounts to SKUs using the SKU percentage
        self.forecast_sku["Allocated_SKU_Sales"] = (
            self.forecast_sku["Amount__c"] * self.forecast_sku["SKU_Percentage"]
        )

        # Handle cases where there is no historical data
        self.forecast_sku["Allocated_SKU_Sales"] = self.forecast_sku[
            "Allocated_SKU_Sales"
        ].fillna(0)
        self.forecast_sku.drop(
            columns=[
                "Account__r.Name",
                "Account__r.Region__c",
                "CreatedDate",
                "CreatedById",
                "CreatedBy.Name",
                "Month",
            ],
            inplace=True,
        )

        return self.forecast_sku

    def evaluate(
        self,
        forecast_start: str = "2023-06-01",
        forecast_end: str = "2024-07-01",
    ):
        forecast_sku = self.forecast_sku
        forecast_start, forecast_end = [forecast_start, forecast_end]
        actual_sales = self.actual_data.query(
            "Date >= @forecast_start & Date < @forecast_end"
        ).copy()

        actual_sales["Year-Month"] = actual_sales["Date"].dt.to_period("M")

        actual_sku_sales_totals = (
            actual_sales.groupby(
                [
                    "Account Id",
                    "BL Short",
                    "Product Family",
                    "Year-Month",
                    "Product Id",
                ]
            )["Total Sales"]
            .sum()
            .reset_index()
        )

        forecast_sku["Year-Month"] = forecast_sku["Date__c"].dt.to_period("M")

        self.forecast_sku_actual_sales = pd.merge(
            forecast_sku,
            actual_sku_sales_totals,
            on=[
                "Account Id",
                "BL Short",
                "Product Family",
                "Year-Month",
                "Product Id",
            ],
            how="left",
            suffixes=("_forecast", "_actual"),
        )

        self.forecast_sku_actual_sales["Error"] = (
            self.forecast_sku_actual_sales["Allocated_SKU_Sales"]
            - self.forecast_sku_actual_sales["Total Sales"]
        )

        return self.forecast_sku_actual_sales

    def save(self, filename: str):
        with pd.ExcelWriter(filename) as writer:
            self.actual_data.to_excel(writer, sheet_name="actual_data")
            self.forecast_data.to_excel(writer, sheet_name="forecast_data")
            self.forecast_sku.to_excel(writer, sheet_name="forecast_sku")
            self.forecast_sku_actual_sales.to_excel(
                writer, sheet_name="forecast_sku_actual_sales"
            )
