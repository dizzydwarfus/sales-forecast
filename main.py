# Third-Party Imports
import pandas as pd

# Built-in Imports
import sys
import os

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

# Internal Imports
from model.fc_ratio_model import ForecastRatioModel
from model.prophet_model import SalesModel
from model.tables_creator import TablesCreator
from model.plotter import Plotter
from utils.utils import get_sales_data
from utils.access_token import AccessToken
from utils._constants import (
    PAYLOAD,
    DOMAIN,
    API_VERSION,
)

FOLDER_PATH = r"C:\Users\Lian.Zhen-Yang\BASF 3D Printing Solutions GmbH\Revenue Operations - General\06. Requests\SNOP FC Breakdown"
SALES_PATH = r"SCM Forecast - Sales Topic_Summary.csv"
sales_data = get_sales_data(os.path.join(FOLDER_PATH, SALES_PATH))

# # Auth Setup
# auth = AccessToken(domain=DOMAIN, payload=PAYLOAD)
# auth.generate_access_token()
# auth_header = {
#     "Content-Type": "application/json",
#     "Authorization": f"Bearer {auth.access_token}",
# }

# # Salesforce Queries
# api_endpoint = DOMAIN + f"/services/data/v{API_VERSION}/query/?q="

# FORECAST_QUERY = """SELECT
# Account__c, Account__r.Name, CreatedDate, Date__c, Amount__c, CreatedById, CreatedBy.Name, Account__r.Region__c, CurrencyIsoCode, Business_line__c, Product_Family__c
# FROM Forecast__c
# WHERE Account__c != null
# """
# FORECAST_QUERY = FORECAST_QUERY.replace("\n", "").replace(" ", "+").strip()

# # Forecast with Saleesforce Forecasts Model
# fc_model = ForecastRatioModel(
#     sales_data=sales_data,
#     domain=DOMAIN,
#     auth_header=auth_header,
#     api_endpoint=api_endpoint,
#     forecast_query=FORECAST_QUERY,
# )
# fc_model.forecast(
#     train_start="2020-01-01",
#     train_end="2023-06-01",
#     forecast_start="2023-06-01",
#     forecast_end="2024-07-01",
# )
# fc_model.evaluate(
#     forecast_start="2023-06-01",
#     forecast_end="2024-07-01",
# )
# fc_model.save(os.path.join(FOLDER_PATH, "forecast_ratio_model.xlsx"))

# Forecast with Prophet Model - Only for a single dimension
## Need to loop through all combinations of dimensions and aggregate to get full forecast over a period
ALL_DIMENSIONS = (
    sales_data[["Region", "BL Short", "Product Family", "Product Subfamily", "Channel"]]
    .drop_duplicates()
    .dropna()
)
dimensions_loop = ALL_DIMENSIONS.to_dict(orient="records")

FORECAST_PLOTS_PATH = os.path.join(FOLDER_PATH, "forecast_plots")

all_sales_data = []
all_grouped_sales_data = []
all_resampled_sales_data = []
all_forecasted_sales_data = []
all_sales_perf_metrics_data = []
all_forecast_and_actuals_data = []
all_metrics_data = []
all_forecast_tables = []
all_summary_tables = []
failed_dimensions = []

for dimension in dimensions_loop[:11]:
    try:
        sales_model = SalesModel(
            sales_data=sales_data,
            region=dimension.get("Region"),
            business_line=dimension.get("BL Short"),
            product_family=dimension.get("Product Family"),
            product_subfamily=dimension.get("Product Subfamily"),
            sales_channel=dimension.get("Channel"),
            seasonality_mode="multiplicative",
        )
        # sales_model.create_holidays(
        #     country_list=[
        #         "US",
        #         "CA",
        #         "CN",
        #         "HK",
        #         "DE",
        #         "NL",
        #         "FR",
        #         "UK",
        #         "IT",
        #         "ES",
        #         "BE",
        #         "PL",
        #         "CZ",
        #         "AT",
        #         "CH",
        #         "SE",
        #         "FI",
        #         "NO",
        #         "DK",
        #         "PT",
        #         "IE",
        #         "GR",
        #         "HU",
        #         "SK",
        #         "RO",
        #         "BG",
        #         "HR",
        #         "SI",
        #         "LT",
        #         "LV",
        #         "EE",
        #         "LU",
        #         "MT",
        #         "CY",
        #     ]
        # )
        sales_model.train()
        sales_model.forecast(
            future_periods=365,
            freq="D",
            include_history=True,
        )
        sales_model.evaluate(
            initial="730 days",
            period="184 days",
            horizon="365 days",
        )

        tables_creator = TablesCreator(
            filtered_sales_data=sales_model.data,
            grouped_sales_data=sales_model.grouped_data,
            resampled_sales_data=sales_model.resampled_data,
            forecasted_sales_data=sales_model.forecast_df,
            evaluation_period="month",
        )

        plotter = Plotter(
            sales_model=sales_model,
            tables_creator=tables_creator,
            evaluation_period="month",
        )
        plotter.plot(forecast_plots_path=FORECAST_PLOTS_PATH)

        # label all data with dimension
        def label_tables(data: pd.DataFrame, dimension: dict):
            data["Region"] = dimension.get("Region")
            data["Business Line"] = dimension.get("BL Short")
            data["Product Family"] = dimension.get("Product Family")
            data["Product Subfamily"] = dimension.get("Product Subfamily")
            data["Channel"] = dimension.get("Channel")
            return data

        all_sales_data.append(label_tables(tables_creator.sales_data, dimension))
        all_grouped_sales_data.append(
            label_tables(tables_creator.grouped_sales_data, dimension)
        )
        all_resampled_sales_data.append(
            label_tables(tables_creator.resampled_sales_data, dimension)
        )
        all_forecasted_sales_data.append(
            label_tables(tables_creator.forecasted_sales_data, dimension)
        )
        all_sales_perf_metrics_data.append(
            label_tables(sales_model.perf_metrics, dimension)
        )
        all_forecast_and_actuals_data.append(
            label_tables(tables_creator.forecast_and_actuals, dimension)
        )
        all_metrics_data.append(
            label_tables(pd.DataFrame(tables_creator.metrics_data), dimension)
        )
        all_forecast_tables.append(
            label_tables(tables_creator.forecast_table, dimension)
        )
        all_summary_tables.append(label_tables(tables_creator.summary_table, dimension))

    except Exception as e:
        print(f"Error: {e}")
        failed_dimensions.append(dimension)
        continue

# combine all data
all_sales_data = pd.concat(all_sales_data)
all_grouped_sales_data = pd.concat(all_grouped_sales_data)
all_resampled_sales_data = pd.concat(all_resampled_sales_data)
all_forecasted_sales_data = pd.concat(all_forecasted_sales_data)
all_sales_perf_metrics_data = pd.concat(all_sales_perf_metrics_data)
all_forecast_and_actuals_data = pd.concat(all_forecast_and_actuals_data)
all_metrics_data = pd.concat(all_metrics_data)
all_forecast_tables = pd.concat(all_forecast_tables)
all_summary_tables = pd.concat(all_summary_tables)
all_failed_dimensions = pd.DataFrame(failed_dimensions)


with pd.ExcelWriter(os.path.join(FOLDER_PATH, "prophet_model.xlsx")) as writer:
    # all_sales_data.to_excel(writer, sheet_name="raw_sales", index=False)
    # all_grouped_sales_data.to_excel(writer, sheet_name="grouped_sales", index=False)
    # all_resampled_sales_data.to_excel(writer, sheet_name="resampled_sales", index=False)
    all_forecasted_sales_data.to_excel(
        writer, sheet_name="forecasted_sales", index=False
    )
    # all_forecast_and_actuals_data.to_excel(
    #     writer, sheet_name="forecast_and_actuals", index=False
    # )
    all_forecast_tables.to_excel(
        writer, sheet_name="forecast_summary_table", index=False
    )
    all_summary_tables.to_excel(writer, sheet_name="summary_table", index=False)
    # all_sales_perf_metrics_data.to_excel(
    #     writer, sheet_name="performance_metrics", index=False
    # )
    all_metrics_data.to_excel(writer, sheet_name="metrics_data", index=False)
    all_failed_dimensions.to_excel(writer, sheet_name="failed_dimensions", index=False)
