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
    df = pd.read_parquet(filename)
    print(f"[Answer Q3] Raw data records count (未篩選): {len(df)}")
    return df


def read_dataframe_filtered(filename):
    df = pd.read_parquet(filename)
    df["duration"] = (
        df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    ).dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    print(f"[Answer Q4] Filtered data records count (duration 1~60 min): {len(df)}")
    return df


def split_data(df, train_ratio=0.8):
    train_size = int(len(df) * train_ratio)
    df_train = df.iloc[:train_size].copy()
    df_val = df.iloc[train_size:].copy()
    return df_train, df_val


def prepare_features(df, dv=None, fit_dv=True):
    features = df[["PULocationID", "DOLocationID"]]
    dicts = features.to_dict(orient="records")

    if fit_dv:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv


def train_and_evaluate(df_train, df_val):
    X_train, dv = prepare_features(df_train, fit_dv=True)
    X_val, _ = prepare_features(df_val, dv=dv, fit_dv=False)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_val)

    rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))

    return lr, dv, rmse


def save_artifacts(dv, lr):
    os.makedirs("models", exist_ok=True)

    with open("models/dv.pkl", "wb") as f_out:
        pickle.dump(dv, f_out)

    with open("models/model.pkl", "wb") as f_out:
        pickle.dump(lr, f_out)


def mlflow_log(rmse):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("yellow-taxi-2023-03")

    with mlflow.start_run():
        mlflow.log_metric("rmse", rmse)
        mlflow.log_params({"model": "LinearRegression"})

        mlflow.log_artifact("models/dv.pkl", artifact_path="preprocessor")
        mlflow.log_artifact("models/model.pkl", artifact_path="model")


def pipeline():
    df_raw = read_raw_dataframe(yellow_taxi_url)
    df = read_dataframe_filtered(yellow_taxi_url)
    df_train, df_val = split_data(df)

    lr, dv, rmse = train_and_evaluate(df_train, df_val)
    save_artifacts(dv, lr)
    mlflow_log(rmse)

    print(f"[Answer Q5] Model intercept: {lr.intercept_:.2f}")
    model_size = os.path.getsize("models/model.pkl")
    print(f"[Answer Q6] Model size in bytes: {model_size}")
    print("[Done]")


if __name__ == "__main__":
    pipeline()
