from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import os
import logging
from yellow_taxi_pipeline import (
    read_raw_dataframe,
    split_data,
    train_and_evaluate,
)

DATA_DIR = "/opt/airflow/data"

default_args = {
    "owner": "小安",
    "depends_on_past": False,
    "start_date": datetime(2025, 6, 3),
    "retries": 1,
}

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

with DAG(
    dag_id="yellow_taxi_pipeline",
    schedule_interval=None,
    default_args=default_args,
    catchup=False,
) as dag:

    def task_read_raw(**kwargs):
        ensure_data_dir()
        logging.info("開始讀取原始資料")
        df_raw = read_raw_dataframe()
        raw_path = os.path.join(DATA_DIR, "df_raw.parquet")
        df_raw.to_parquet(raw_path)
        logging.info(f"原始資料儲存完成：{raw_path}")
        return raw_path  # 自動推到 XCom，key 為 return_value

    def task_read_filtered(ti):
        raw_path = ti.xcom_pull(task_ids="task_read_raw")
        logging.info(f"從路徑載入原始資料：{raw_path}")
        df_raw = pd.read_parquet(raw_path)

        # 過濾邏輯直接寫這裡，避免重新下載
        df_raw["duration"] = (
            df_raw.tpep_dropoff_datetime - df_raw.tpep_pickup_datetime
        ).dt.total_seconds() / 60
        df_filtered = df_raw[(df_raw["duration"] >= 1) & (df_raw["duration"] <= 60)]

        categorical = ["PULocationID", "DOLocationID"]
        df_filtered[categorical] = df_filtered[categorical].astype(str)

        filtered_path = os.path.join(DATA_DIR, "df_filtered.parquet")
        df_filtered.to_parquet(filtered_path)
        logging.info(f"過濾後資料儲存完成：{filtered_path}")
        return filtered_path

    def task_split(ti):
        filtered_path = ti.xcom_pull(task_ids="task_read_filtered")
        logging.info(f"載入過濾資料：{filtered_path}")
        df_filtered = pd.read_parquet(filtered_path)
        df_train, df_val = split_data(df_filtered)

        train_path = os.path.join(DATA_DIR, "df_train.parquet")
        val_path = os.path.join(DATA_DIR, "df_val.parquet")
        df_train.to_parquet(train_path)
        df_val.to_parquet(val_path)
        logging.info(f"訓練與驗證資料已分割並儲存：{train_path}, {val_path}")
        return {"train_path": train_path, "val_path": val_path}

    def task_train(ti):
        import logging
        import traceback
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        try:
            paths = ti.xcom_pull(task_ids="task_split")
            train_path = paths["train_path"]
            val_path = paths["val_path"]
            logger.info(f"載入訓練資料：{train_path}")
            logger.info(f"載入驗證資料：{val_path}")

            df_train = pd.read_parquet(train_path)
            df_val = pd.read_parquet(val_path)

            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
            logger.info(f"開始模型訓練，MLflow 追蹤 URI: {tracking_uri}")

            rmse = train_and_evaluate(df_train, df_val, tracking_uri=tracking_uri, logger=logger)
            logger.info(f"模型訓練完成，RMSE: {rmse}")

            return rmse

        except Exception as e:
            logger.error(f"[Train Task] 執行失敗：{e}")
            logger.error(traceback.format_exc())
            raise


    read_raw = PythonOperator(
        task_id="task_read_raw",
        python_callable=task_read_raw,
    )

    read_filtered = PythonOperator(
        task_id="task_read_filtered",
        python_callable=task_read_filtered,
    )

    split_data_task = PythonOperator(
        task_id="task_split",
        python_callable=task_split,
    )

    train_task = PythonOperator(
        task_id="task_train",
        python_callable=task_train,
    )

    read_raw >> read_filtered >> split_data_task >> train_task
