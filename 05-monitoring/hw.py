import datetime
import time
import random
import logging
import pandas as pd
import joblib
import psycopg

from prefect import task, flow
from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnQuantileMetric,
)

# 設定 logging 格式
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

SEND_TIMEOUT = 10
rand = random.Random()

# 建立 PostgreSQL 資料表
create_table_statement = """
drop table if exists homework;
create table homework(
    timestamp timestamp,
    prediction_drift float,
    num_drifted_columns integer,
    share_missing_values float,
    fare_mean float,
    fare_median float
)
"""

# 讀取參考資料與模型
reference_data = pd.read_parquet("data/reference.parquet", engine="fastparquet")
raw_data = pd.read_parquet("data/green_tripdata_2024-03.parquet", engine="fastparquet")

with open("models/lin_reg.bin", "rb") as f_in:
    model = joblib.load(f_in)

begin = datetime.datetime(2024, 3, 1, 0, 0)

# 特徵定義
num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
cat_features = ["PULocationID", "DOLocationID"]

column_mapping = ColumnMapping(
    prediction="prediction",
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None,
)

# 設定 Evidently metrics（含中位數）
report = Report(
    metrics=[
        ColumnDriftMetric(column_name="prediction"),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        ColumnQuantileMetric(column_name="fare_amount", quantile=0.5),
    ]
)

@task
def prep_db():
    with psycopg.connect(
        "host=localhost port=5432 user=postgres password=example", autocommit=True
    ) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
        with psycopg.connect(
            "host=localhost port=5432 dbname=test user=postgres password=example"
        ) as conn:
            conn.execute(create_table_statement)

@task
def calculate_metrics_postgresql(i: int):
    current_data = raw_data[
        (raw_data.lpep_pickup_datetime >= (begin + datetime.timedelta(i)))
        & (raw_data.lpep_pickup_datetime < (begin + datetime.timedelta(i + 1)))
    ].copy()

    # 填補空值，避免模型預測錯誤
    current_data[num_features + cat_features] = current_data[num_features + cat_features].fillna(0)
    current_data["prediction"] = model.predict(current_data[num_features + cat_features])

    # 執行 Evidently report
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )
    result = report.as_dict()

    # 從 Evidently 結果中取得指標
    prediction_drift = result["metrics"][0]["result"]["drift_score"]
    num_drifted_columns = result["metrics"][1]["result"]["number_of_drifted_columns"]
    share_missing_values = result["metrics"][2]["result"]["current"]["share_of_missing_values"]
    fare_median = result["metrics"][3]["result"]["current"]['value']
    # 安全取得 fare_median
    # fare_median = None
    

    # 使用 pandas 計算平均數
    fare_mean = current_data["fare_amount"].mean()

    logging.info(
        f"{i+1:02d}日 ➜ drift={prediction_drift:.3f} | drifted_cols={num_drifted_columns} | "
        f"missing={share_missing_values:.3f} | fare_mean={fare_mean:.2f} | fare_median={fare_median:.2f}"
    )

    # 寫入 PostgreSQL
    with psycopg.connect(
        "host=localhost port=5432 dbname=test user=postgres password=example",
        autocommit=True,
    ) as conn:
        with conn.cursor() as curr:
            curr.execute(
                """
                insert into homework(
                    timestamp, prediction_drift, num_drifted_columns,
                    share_missing_values, fare_mean, fare_median
                )
                values (%s, %s, %s, %s, %s, %s)
                """,
                (
                    begin + datetime.timedelta(i),
                    prediction_drift,
                    num_drifted_columns,
                    share_missing_values,
                    fare_mean,
                    fare_median,
                ),
            )

@flow
def batch_monitoring_backfill():
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)

    for i in range(0, 31):
        calculate_metrics_postgresql(i)

        new_send = datetime.datetime.now()
        seconds_elapsed = (new_send - last_send).total_seconds()

        if seconds_elapsed < SEND_TIMEOUT:
            time.sleep(SEND_TIMEOUT - seconds_elapsed)

        while last_send < new_send:
            last_send += datetime.timedelta(seconds=10)

        logging.info("data sent")

if __name__ == "__main__":
    batch_monitoring_backfill()


# -- 查看中位數最大值
# SELECT MAX(fare_median) FROM homework;