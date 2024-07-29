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

# Auth Setup
auth = AccessToken(domain=DOMAIN, payload=PAYLOAD)
auth.generate_access_token()
auth_header = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {auth.access_token}",
}

# Salesforce Queries
api_endpoint = DOMAIN + f"/services/data/v{API_VERSION}/query/?q="

FORECAST_QUERY = """SELECT 
Account__c, Account__r.Name, CreatedDate, Date__c, Amount__c, CreatedById, CreatedBy.Name, Account__r.Region__c, CurrencyIsoCode, Business_line__c, Product_Family__c 
FROM Forecast__c 
WHERE Account__c != null
"""
FORECAST_QUERY = FORECAST_QUERY.replace("\n", "").replace(" ", "+").strip()

# Forecast with Saleesforce Forecasts Model
fc_model = ForecastRatioModel(
    sales_path=os.path.join(FOLDER_PATH, SALES_PATH),
    domain=DOMAIN,
    auth_header=auth_header,
    api_endpoint=api_endpoint,
    forecast_query=FORECAST_QUERY,
)
fc_model.forecast(
    train_start="2020-01-01",
    train_end="2023-06-01",
    forecast_start="2023-06-01",
    forecast_end="2024-07-01",
)
fc_model.evaluate(
    forecast_start="2023-06-01",
    forecast_end="2024-07-01",
)
fc_model.save(os.path.join(FOLDER_PATH, "forecast_ratio_model.xlsx"))

# Forecast with Prophet Model - Only for a single dimension
## Need to loop through all combinations of dimensions and aggregate to get full forecast over a period
sales_data = get_sales_data(os.path.join(FOLDER_PATH, SALES_PATH))
sales_model = SalesModel(
    sales_data=sales_data,
    region="Europe",
    business_line="AES",
    product_family="Standard",
    sales_channel="DISTRIBUTOR & RESELLER",
    seasonality_mode="multiplicative",
)
sales_model.create_holidays(
    country_list=[
        "DE",
        "NL",
        "FR",
        "UK",
        "IT",
        "ES",
        "BE",
        "PL",
        "CZ",
        "AT",
        "CH",
        "SE",
        "FI",
        "NO",
        "DK",
        "PT",
        "IE",
        "GR",
        "HU",
        "SK",
        "RO",
        "BG",
        "HR",
        "SI",
        "LT",
        "LV",
        "EE",
        "LU",
        "MT",
        "CY",
    ]
)
sales_model.train()
sales_model.forecast()
sales_model.evaluate()

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
plotter.plot()

with pd.ExcelWriter(os.path.join(FOLDER_PATH, "prophet_model.xlsx")) as writer:
    tables_creator.sales_data.to_excel(writer, sheet_name="raw_sales")
    tables_creator.grouped_sales_data.to_excel(writer, sheet_name="grouped_sales")
    tables_creator.resampled_sales_data.to_excel(writer, sheet_name="resampled_sales")
    tables_creator.forecasted_sales_data.to_excel(writer, sheet_name="forecasted_sales")
    tables_creator.forecast_and_actuals.to_excel(
        writer, sheet_name="forecast_and_actuals"
    )
    tables_creator.forecast_table.to_excel(writer, sheet_name="forecast_summary_table")
    tables_creator.summary_table.to_excel(writer, sheet_name="summary_table")
    sales_model.perf_metrics.to_excel(writer, sheet_name="performance_metrics")
    pd.DataFrame(tables_creator.metrics_data).to_excel(
        writer, sheet_name="metrics_data"
    )
