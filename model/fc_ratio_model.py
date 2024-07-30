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
        self.sku_sales_totals = None
        self.group_sales_totals = None
        self.sku_sales_percentage = None
        self.filtered_forecast = None
        self.forecast_sku = None
        self.forecast_sku_actual_sales = None

    def _get_forecast_data(
        self,
    ):
        self.forecast_data = get_forecast_data(
            domain=self.domain,
            api_endpoint=self.api_endpoint,
            auth_header=self.auth_header,
            forecast_query=self.forecast_query,
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

        # remove negative sales
        historical_sales = historical_sales[historical_sales["Total Sales"] >= 0]

        # use forecast data for the next year (this determines the forecast period)
        self.filtered_forecast = self.forecast_data.query(
            "Date__c >= @forecast_start & Date__c < @forecast_end"
        ).copy()

        # Extract month and year for grouping
        historical_sales["Month"] = historical_sales["Date"].dt.month_name()
        self.filtered_forecast["Month"] = self.filtered_forecast[
            "Date__c"
        ].dt.month_name()

        # create year-month column for grouping
        historical_sales["Year-Month"] = historical_sales["Date"].dt.to_period("M")
        self.filtered_forecast["Year-Month"] = self.filtered_forecast[
            "Date__c"
        ].dt.to_period("M")

        # Calculate total sales per SKU per grouping
        # grouping by Month to take long-term trends into account
        self.sku_sales_totals = (
            historical_sales.groupby(
                [
                    "Account Id",
                    # "Account Name",
                    "Region",
                    "Channel",
                    "BL Short",
                    "Product Family",
                    "Product Subfamily",
                    "Product Id",
                    "Local Item Code",
                    "Local Item Description",
                    "Month",
                ]
            )["Total Sales"]
            .sum()
            .reset_index()
        )

        self.group_sales_totals = (
            historical_sales.groupby(
                [
                    "Account Id",
                    # "Account Name",
                    "Region",
                    "Channel",
                    "BL Short",
                    "Product Family",
                    # "Product Subfamily",
                    "Month",
                ]
            )["Total Sales"]
            .sum()
            .reset_index()
        )

        # Merge to calculate SKU percentage
        self.sku_sales_percentage = pd.merge(
            self.sku_sales_totals,
            self.group_sales_totals,
            on=[
                "Account Id",
                "Region",
                "Channel",
                "BL Short",
                "Product Family",
                # "Product Subfamily",
                "Month",
            ],
            suffixes=("_sku", "_total"),
        )
        self.sku_sales_percentage["SKU_Percentage"] = (
            self.sku_sales_percentage["Total Sales_sku"]
            / self.sku_sales_percentage["Total Sales_total"]
        )

        # Merge forecast data with SKU percentage distribution
        self.filtered_forecast = self.filtered_forecast.rename(
            columns={
                "Account__c": "Account Id",
                "Account__r.Name": "Account Name",
                "Business_line__c": "BL Short",
                "Product_Family__c": "Product Family",
                "Account__r.Region__c": "Region",
                "Account__r.Channel__c": "Channel",
            }
        )

        self.filtered_forecast = (
            self.filtered_forecast.groupby(
                [
                    "Account Id",
                    "Account Name",
                    "Region",
                    "Channel",
                    "BL Short",
                    "Product Family",
                    "Year-Month",
                    "Month",
                ]
            )["Amount__c"]
            .sum()
            .reset_index()
        )
        self.filtered_forecast["Channel"] = self.filtered_forecast[
            "Channel"
        ].str.upper()

        self.forecast_sku = pd.merge(
            self.filtered_forecast,
            self.sku_sales_percentage,
            on=[
                "Account Id",
                "Region",
                "Channel",
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
                    "Account Name",
                    "Region",
                    "Channel",
                    "BL Short",
                    "Product Family",
                    "Product Subfamily",
                    "Product Id",
                    "Local Item Code",
                    "Local Item Description",
                    "Year-Month",
                ],
            )["Total Sales"]
            .sum()
            .reset_index()
        )

        self.forecast_sku_actual_sales = pd.merge(
            forecast_sku,
            actual_sku_sales_totals,
            on=[
                "Account Id",
                "Region",
                "Channel",
                "BL Short",
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
            self.actual_data.to_excel(writer, sheet_name="actual_data", index=False)
            self.forecast_data.to_excel(writer, sheet_name="forecast_data", index=False)
            self.forecast_sku.to_excel(writer, sheet_name="forecast_sku", index=False)
            self.forecast_sku_actual_sales.to_excel(
                writer, sheet_name="forecast_sku_actual_sales", index=False
            )
