import pandas as pd
import numpy as np
import requests
from utils.access_token import AccessToken
import re


def sanitize_filename(filename):
    # Replace any character that is not alphanumeric, underscore, or hyphen with an underscore
    return re.sub(r"[^\w\-]", "_", filename)


def generate_auth_header(domain: str, payload: dict) -> dict:
    auth = AccessToken(domain=domain, payload=payload)
    auth.generate_access_token()
    auth_header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth.access_token}",
    }
    return auth_header


def flatten_dictionary(dict_to_flatten):
    new_dict = {}
    values_data_type = []

    for _, value_type in enumerate(dict_to_flatten):
        values_data_type.append(isinstance(dict_to_flatten[value_type], dict))

    if True not in values_data_type:
        return dict_to_flatten

    for _, lvl1 in enumerate(dict_to_flatten):
        if isinstance(dict_to_flatten[lvl1], dict):
            for _, lvl2 in enumerate(dict_to_flatten[lvl1]):
                new_dict[lvl1 + "." + lvl2] = dict_to_flatten[lvl1][lvl2]
        else:
            new_dict[lvl1] = dict_to_flatten[lvl1]

    return flatten_dictionary(new_dict)


def format_query(query: str) -> str:
    query = query.replace("\n", "").replace(" ", "+").strip()
    return query


def get_data(
    domain: str, api_endpoint: str, query: str, auth_header: dict, **kwargs
) -> pd.DataFrame:
    all_data = []

    response = requests.get(url=api_endpoint + query, headers=auth_header)
    data = response.json()
    all_data.extend(data["records"])

    while "nextRecordsUrl" in data:
        response = requests.get(
            url=domain + data["nextRecordsUrl"], headers=auth_header
        )
        data = response.json()

        all_data.extend(data["records"])

    for i, record in enumerate(all_data):
        del record["attributes"]

        if len(kwargs) == 0:
            pass

        for _, v in kwargs.items():
            if record.get(v):
                del record[v]["attributes"]

        all_data[i] = flatten_dictionary(record)

    all_data = pd.DataFrame(all_data)
    return all_data


def format_euro(value):
    return f"€{value:,.0f}" if value >= 0 else f"-€{abs(value):,.0f}"


def get_forecast_data(
    domain: str, api_endpoint: str, auth_header: dict
) -> pd.DataFrame:
    forecast_query = """SELECT 
    Account__c, Account__r.Name, CreatedDate, Date__c, Amount__c, CreatedById, CreatedBy.Name, Account__r.Region__c, CurrencyIsoCode, Business_line__c, Product_Family__c 
    FROM Forecast__c 
    WHERE Account__c != null
    """
    forecast_query = forecast_query.replace("\n", "").replace(" ", "+").strip()

    raw_forecast_data = get_data(
        domain=domain,
        api_endpoint=api_endpoint,
        query=forecast_query,
        auth_header=auth_header,
        val1="Account__r",
        val2="CreatedBy",
    )

    raw_forecast_data["CreatedDate"] = pd.to_datetime(
        raw_forecast_data["CreatedDate"], format="%Y-%m-%dT%H:%M:%S.000+0000"
    )

    raw_forecast_data["Date__c"] = pd.to_datetime(
        raw_forecast_data["Date__c"], format="%Y-%m-%d"
    )

    raw_forecast_data = raw_forecast_data.dropna(subset=["Product_Family__c"])

    return raw_forecast_data


def get_sales_data(sales_path: str) -> pd.DataFrame:
    sales_data = pd.read_csv(
        filepath_or_buffer=sales_path, date_format="%d-%b-%y", parse_dates=["Date"]
    )
    sales_data = sales_data.pivot(
        values="Measure Values",
        columns=["Measure Names"],
        index=[
            "Account Id",
            "Account Name",
            "Business Line",
            "Region",
            "Channel",
            "Product Family (Account Hierarchy)",
            "Product Subfamily (Account Hierarchy)",
            "Product Id",
            "Local Item Code (Zita)",
            "Local Item Description (Zita)",
            "Date",
        ],
    ).reset_index()
    sales_data = sales_data.rename(
        columns={
            "Product Family (Account Hierarchy)": "Product Family",
            "Product Subfamily (Account Hierarchy)": "Product Subfamily",
            "Local Item Code (Zita)": "Local Item Code",
            "Local Item Description (Zita)": "Local Item Description",
        }
    )

    sales_data["Business Line"] = (
        sales_data["Business Line"].replace(np.nan, "Unallocated (Unallocated)").copy()
    )
    sales_data["BL Short"] = (
        sales_data["Business Line"].str.findall(r"\((.*?)\)").str[0].copy()
    )

    return sales_data
