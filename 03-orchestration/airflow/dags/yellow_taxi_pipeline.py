import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow
import numpy as np
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# 資料來源
yellow_taxi_url = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
)

# 讀取原始資料（不過濾）
def read_raw_dataframe():
    df = pd.read_parquet(yellow_taxi_url)
    return df

# 讀取並過濾資料（duration 1~60 分）
def read_filtered_dataframe():
    df = pd.read_parquet(yellow_taxi_url)
    df["duration"] = (
        df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    ).dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    return df

# 分割訓練與驗證資料
def split_data(df):
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size].copy()
    df_val = df.iloc[train_size:].copy()
    return df_train, df_val

# 特徵轉換工具
def prepare_features(df, dv=None, fit_dv=True):
    features = df[["PULocationID", "DOLocationID"]]
    dicts = features.to_dict(orient="records")

    if fit_dv:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv

# 模型訓練 + 回傳 RMSE
def train_and_evaluate(df_train, df_val, tracking_uri="http://localhost:5000", logger=None):
    import mlflow
    import pickle
    import os
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.metrics import mean_absolute_error, r2_score
    from datetime import datetime
    import subprocess

    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("yellow-taxi-2023-03")

    start_time = datetime.now()

    # 紀錄 git commit
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        mlflow.set_tag("git_commit", git_commit)
    except Exception:
        pass

    X_train, dv = prepare_features(df_train, fit_dv=True)
    X_val, _ = prepare_features(df_val, dv=dv, fit_dv=False)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    with mlflow.start_run(run_name=f"run-{start_time.isoformat()}"):
        try:
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_val)

            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            mlflow.log_params({
                "model": "LinearRegression",
                "num_features": X_train.shape[1],
                "num_train_samples": len(df_train),
                "num_val_samples": len(df_val),
            })

            mlflow.set_tag("model_type", "LinearRegression")
            mlflow.set_tag("env", "production")

            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)

            dv_path = os.path.join(model_dir, "dv.pkl")
            model_path = os.path.join(model_dir, "model.pkl")

            with open(dv_path, "wb") as f_out:
                pickle.dump(dv, f_out)
            mlflow.log_artifact(dv_path, artifact_path="preprocessor")

            with open(model_path, "wb") as f_out:
                pickle.dump(lr, f_out)
            mlflow.log_artifact(model_path, artifact_path="model")

            end_time = datetime.now()
            mlflow.log_param("training_start_time", start_time.isoformat())
            mlflow.log_param("training_end_time", end_time.isoformat())
            mlflow.log_param("training_duration_sec", (end_time - start_time).total_seconds())

            logger.info(f"[MLflow] RMSE: {rmse:.2f}")
            logger.info(f"[MLflow] Model saved at: {model_path}")

        except Exception as e:
            mlflow.log_param("error", str(e))
            logger.error(f"訓練過程出錯: {e}")
            raise

    return rmse
