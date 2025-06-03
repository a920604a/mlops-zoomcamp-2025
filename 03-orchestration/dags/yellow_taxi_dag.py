# file: dags/yellow_taxi_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys

sys.path.insert(0, "/opt/airflow/dags")  # 確保可找到 pipeline 檔案

from yellow_taxi_pipeline import (
    read_raw_dataframe,
    read_filtered_dataframe,
    split_data,
    train_and_evaluate,
)

default_args = {
    "owner": "小安",
    "depends_on_past": False,
    "start_date": datetime(2025, 6, 3),
    "retries": 1,
}

with DAG(
    "yellow_taxi_pipeline",
    schedule_interval=None,  # 手動觸發或改成你想的排程
    default_args=default_args,
    catchup=False,
) as dag:

    def task_read_raw(**kwargs):
        df_raw = read_raw_dataframe()
        kwargs["ti"].xcom_push(key="df_raw", value=df_raw.to_json())

    def task_read_filtered(**kwargs):
        df_filtered = read_filtered_dataframe()
        kwargs["ti"].xcom_push(key="df_filtered", value=df_filtered.to_json())

    def task_split(**kwargs):
        ti = kwargs["ti"]
        df_filtered_json = ti.xcom_pull(key="df_filtered", task_ids="read_filtered")
        df_filtered = pd.read_json(df_filtered_json)
        df_train, df_val = split_data(df_filtered)
        ti.xcom_push(key="df_train", value=df_train.to_json())
        ti.xcom_push(key="df_val", value=df_val.to_json())

    def task_train(**kwargs):
        ti = kwargs["ti"]
        df_train = pd.read_json(ti.xcom_pull(key="df_train", task_ids="split_data"))
        df_val = pd.read_json(ti.xcom_pull(key="df_val", task_ids="split_data"))
        rmse = train_and_evaluate(df_train, df_val)
        print(f"Training done with RMSE: {rmse}")

    read_raw = PythonOperator(
        task_id="read_raw",
        python_callable=task_read_raw,
        provide_context=True,
    )

    read_filtered = PythonOperator(
        task_id="read_filtered",
        python_callable=task_read_filtered,
        provide_context=True,
    )

    split_data_task = PythonOperator(
        task_id="split_data",
        python_callable=task_split,
        provide_context=True,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=task_train,
        provide_context=True,
    )

    # 定義 Task 執行順序
    read_raw >> read_filtered >> split_data_task >> train_task
