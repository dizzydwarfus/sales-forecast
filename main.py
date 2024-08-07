# %%
# Third-Party Imports
import pandas as pd
from dotenv import load_dotenv

# Built-in Imports
import sys
import os
import traceback

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
from utils.logger import MyLogger

load_dotenv()

# Logger Setup
current_filename = os.path.basename(__file__)
logger = MyLogger(name=current_filename).get_logger()

FOLDER_PATH = os.getenv("SNOP_BREAKDOWN_PATH")
SALES_PATH = r"data\SCM Forecast - Sales Topic_Summary.csv"
sales_data = get_sales_data(os.path.join(FOLDER_PATH, SALES_PATH))

# Auth Setup
try:
    auth = AccessToken(domain=DOMAIN, payload=PAYLOAD)
    auth.generate_access_token()
    auth_header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth.access_token}",
    }
except Exception as e:
    logger.error("Failed to generate access token.")
    logger.error(f"{type(e).__name__}: {e}")
    logger.error(traceback.format_exc())

# Salesforce Queries
api_endpoint = DOMAIN + f"/services/data/v{API_VERSION}/query/?q="

FORECAST_QUERY = """SELECT 
Account__c, Account__r.Name, Account__r.Channel__c, CreatedDate, Date__c, Amount__c, CreatedById, CreatedBy.Name, Account__r.Region__c, CurrencyIsoCode, Business_line__c, Product_Family__c 
FROM Forecast__c 
WHERE Account__c != null and Account__r.Name != 'NorthAmerica test'
"""
FORECAST_QUERY = FORECAST_QUERY.replace("\n", "").replace(" ", "+").strip()

# Forecast with Salesforce Forecasts Model
try:
    fc_model = ForecastRatioModel(
        sales_data=sales_data,
        domain=DOMAIN,
        auth_header=auth_header,
        api_endpoint=api_endpoint,
        forecast_query=FORECAST_QUERY,
    )
except Exception as e:
    logger.error("Failed to create ForecastRatioModel.")
    logger.error(f"{type(e).__name__}: {e}")
    logger.error(traceback.format_exc())

try:
    fc_model.forecast(
        train_start="2022-01-01",
        train_end="2023-06-01",
        forecast_start="2023-06-01",
        forecast_end="2024-07-01",
    )
except Exception as e:
    logger.error("Failed to forecast with ForecastRatioModel.")
    logger.error(f"{type(e).__name__}: {e}")
    logger.error(traceback.format_exc())
# %%
try:
    fc_model.evaluate(
        forecast_start="2023-06-01",
        forecast_end="2024-07-01",
    )
except Exception as e:
    logger.error("Failed to evaluate ForecastRatioModel.")
    logger.error(f"{type(e).__name__}: {e}")
    logger.error(traceback.format_exc())

try:
    fc_model.save(
        os.path.join(FOLDER_PATH, "model excels", "forecast_ratio_model.xlsx")
    )
    logger.info(f"Successfully saved forecast_ratio_model.xlsx to {FOLDER_PATH}.")
except Exception as e:
    logger.error("Failed to save ForecastRatioModel.")
    logger.error(f"{type(e).__name__}: {e}")
    logger.error(traceback.format_exc())

# %%
# Forecast with Prophet Model - Only for a single dimension
## Need to loop through all combinations of dimensions and aggregate to get full forecast over a period
agg_sales_data = (
    sales_data.groupby(
        [
            "Region",
            "BL Short",
            "Product Family",
            "Product Subfamily",
            "Product Id",
            "Product Description",
            "Channel",
        ]
    )
    .agg({"Total Sales": ["sum", "count"]})
    .reset_index()
    .sort_values(by=[("Total Sales", "sum")], ascending=False)
)

agg_sales_data.columns = [" ".join(cols).strip() for cols in agg_sales_data.columns]

ALL_DIMENSIONS = (
    agg_sales_data[
        [
            "Region",
            "BL Short",
            "Product Family",
            "Product Subfamily",
            "Product Id",
            "Product Description",
            "Channel",
        ]
    ]
    .drop_duplicates()
    .dropna()
)
dimensions_loop = ALL_DIMENSIONS.to_dict(orient="records")
logger.info(f"Total number of dimensions: {len(dimensions_loop)}")

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

for dimension in dimensions_loop[:100]:
    if dimension.get("BL Short") == "AMS":
        logger.info(f"Skipping dimension: {dimension} because AMS.")
        continue

    if (
        dimension.get("Product Subfamily") == "Other"
        and dimension.get("Product Subfamily") == "Other"
    ):
        logger.info(f"Skipping dimension: {dimension} because Other.")
        continue
    try:
        sales_model = SalesModel(
            sales_data=sales_data,
            region=dimension.get("Region"),
            sales_channel=dimension.get("Channel"),
            business_line=dimension.get("BL Short"),
            product_family=dimension.get("Product Family"),
            product_subfamily=dimension.get("Product Subfamily"),
            product_id=dimension.get("Product Id"),
            product_description=dimension.get("Product Description"),
            prediction_end_date="2024-12-31",
            seasonality_mode="multiplicative",
        )
    except Exception as e:
        logger.error(f"Failed to create SalesModel for {dimension}.")
        logger.error(f"{type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        failed_dimensions.append(dimension)
        continue

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
    try:
        sales_model.train()
        sales_model.forecast(
            freq="D",
            include_history=True,
        )
    except Exception as e:
        logger.error(f"Failed to train and forecast SalesModel for {dimension}.")
        logger.error(f"{type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        failed_dimensions.append(dimension)
        continue

    try:
        sales_model.evaluate(
            initial="365 days",
            period="184 days",
            horizon="184 days",
        )
    except Exception as e:
        logger.error(f"Failed to evaluate SalesModel for {dimension}.")
        logger.error(f"{type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        failed_dimensions.append(dimension)
        continue

    try:
        tables_creator = TablesCreator(
            filtered_sales_data=sales_model.data,
            grouped_sales_data=sales_model.grouped_data,
            resampled_sales_data=sales_model.resampled_data,
            forecasted_sales_data=sales_model.forecast_df,
            evaluation_period="month",
        )
    except Exception as e:
        logger.error(f"Failed to create TablesCreator for {dimension}.")
        logger.error(f"{type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        failed_dimensions.append(dimension)
        continue

    try:
        plotter = Plotter(
            sales_model=sales_model,
            tables_creator=tables_creator,
            evaluation_period="month",
        )
        plotter.plot(show=False, forecast_plots_path=FORECAST_PLOTS_PATH)

    except Exception as e:
        logger.error(f"Failed to plot data for {dimension}.")
        logger.error(f"{type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        failed_dimensions.append(dimension)
        continue

    try:
        # label all data with dimension
        def label_tables(data: pd.DataFrame, dimension: dict):
            data["Region"] = dimension.get("Region")
            data["Business Line"] = dimension.get("BL Short")
            data["Channel"] = dimension.get("Channel")
            data["Product Family"] = dimension.get("Product Family")
            data["Product Subfamily"] = dimension.get("Product Subfamily")
            data["Product Id"] = dimension.get("Product Id")
            data["Product Description"] = dimension.get("Product Description")
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

        logger.info(f"Successfully modelled data with dimension: {dimension}")

    except Exception as e:
        logger.error(f"Failed to label data with {dimension}.")
        logger.error(f"{type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        failed_dimensions.append(dimension)
        continue

# combine all data
try:
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
except Exception as e:
    logger.error("Failed to combine all data.")
    logger.error(f"{type(e).__name__}: {e}")
    logger.error(traceback.format_exc())
    failed_dimensions.append(dimension)

try:
    prophet_model_file_path = os.path.join(
        FOLDER_PATH, "model excels", "prophet_model.xlsx"
    )
    sales_data["month"] = sales_data["Date"].dt.to_period("M")
    with pd.ExcelWriter(prophet_model_file_path) as writer:
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
        all_failed_dimensions.to_excel(
            writer, sheet_name="failed_dimensions", index=False
        )
        sales_data.to_excel(writer, sheet_name="raw_sales", index=False)

    logger.info(f"Successfully saved data to excel: {prophet_model_file_path}")

except Exception as e:
    logger.error(f"Failed to save data to excel: {prophet_model_file_path}")
    logger.error(f"{type(e).__name__}: {e}")
    logger.error(traceback.format_exc())
# %%
