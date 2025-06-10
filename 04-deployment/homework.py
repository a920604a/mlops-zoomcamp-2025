import pickle
import pandas as pd
import os


def read_data(filename, categorical):
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


with open("./homework/model.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)


def predict():

    categorical = ["PULocationID", "DOLocationID"]

    df = read_data(
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet",
        categorical,
    )

    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    # Q1. Notebook
    # What's the standard deviation of the predicted duration for this dataset?
    print(y_pred.std())  # 6.24


# Q2. Preparing the output
def prepare(year=2023, month=3):
    if not os.path.exists("results"):
        os.makedirs("results")
    output_file = f"results/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    categorical = ["PULocationID", "DOLocationID"]

    df = read_data(
        f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet",
        categorical,
    )
    dicts = df[categorical].to_dict(orient="records")
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    # 將預測結果與 ride_id 包裝成 DataFrame
    df_result = pd.DataFrame({"ride_id": df["ride_id"], "predicted_duration": y_pred})

    df_result.to_parquet(output_file, engine="pyarrow", compression=None, index=False)
    # What's the size of the output file?

    file_size_bytes = os.path.getsize(output_file)
    file_size_kb = file_size_bytes / 1024 / 1024

    print(f"檔案大小為：{file_size_bytes} bytes（約 {file_size_kb:.2f} MB）")  # 65.46MB


# Q3
# jupyter nbconvert --to script homework/starter.ipynb

if __name__ == "__main__":
    # predict()
    prepare()
