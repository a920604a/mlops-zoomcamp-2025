import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow
import numpy as np
import os

yellow_taxi_url = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
)


def read_raw_dataframe(filename):
    # 讀取原始資料，未過濾 duration
    df = pd.read_parquet(filename)
    return df


def read_dataframe_filtered(filename):
    # 讀取資料並過濾 duration 1~60 分鐘
    df = pd.read_parquet(filename)
    df["duration"] = (
        df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    ).dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    return df


def prepare_features(df, dv=None, fit_dv=True):
    features = df[["PULocationID", "DOLocationID"]]
    dicts = features.to_dict(orient="records")

    if fit_dv:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv


def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("yellow-taxi-2023-03")

    print("[Step 1] Loading raw data...")
    df_raw = read_raw_dataframe(yellow_taxi_url)
    print(f"[Answer Q3] Raw data records count (未篩選): {len(df_raw)}")

    print("[Step 2] Loading filtered data...")
    df = read_dataframe_filtered(yellow_taxi_url)
    print(f"[Answer Q4] Filtered data records count (duration 1~60 min): {len(df)}")

    # 拆分訓練/驗證資料
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size].copy()
    df_val = df.iloc[train_size:].copy()

    print("[Step 3] Preparing features...")
    X_train, dv = prepare_features(df_train, fit_dv=True)
    X_val, _ = prepare_features(df_val, dv=dv, fit_dv=False)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    print("[Step 4] Training Linear Regression model...")
    with mlflow.start_run():
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_val)

        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))

        mlflow.log_metric("rmse", rmse)
        mlflow.log_params({"model": "LinearRegression"})

        os.makedirs("models", exist_ok=True)
        with open("models/dv.pkl", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/dv.pkl", artifact_path="preprocessor")

        with open("models/model.pkl", "wb") as f_out:
            pickle.dump(lr, f_out)
        mlflow.log_artifact("models/model.pkl", artifact_path="model")

        print(f"[Answer Q5] Model intercept: {lr.intercept_:.2f}")

        model_size = os.path.getsize("models/model.pkl")
        print(f"[Answer Q6] Model size in bytes: {model_size}")

    print("[Done]")


if __name__ == "__main__":
    main()
