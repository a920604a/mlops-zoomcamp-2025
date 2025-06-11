#!/usr/bin/env python
# coding: utf-8


import pickle
import pandas as pd


categorical = ["PULocationID", "DOLocationID"]


def load_model():
    with open("/app/model.bin", "rb") as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def read_data(filename):
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


def fetch_predict_data(model, dv, year=2023, month=3):
    df = read_data(
        f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    )

    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    # What's the mean predicted duration?
    print(f"predicted mean duration: {y_pred.mean()}")  # 0.1917


if __name__ == "__main__":
    dv, model = load_model()
    fetch_predict_data(model, dv, 2023, 5)
