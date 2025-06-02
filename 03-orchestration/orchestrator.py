import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow
import xgboost as xgb
from pathlib import Path

train_url = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-01.parquet"
)
val_url = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-02.parquet"
)


def download_data(year, month, max_retries=3):
    # ingestion
    attempt = 0
    while attempt < max_retries:
        filename = f"green_tripdata_{year}-{month:02d}.parquet"
        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}"
        try:
            df = pd.read_parquet(url)
            if not df.empty:
                return df
            else:
                print(
                    f"Data for {year}-{month:02d} is empty, retrying ({attempt + 1}/{max_retries})..."
                )
        except Exception as e:
            print(f"Failed to download data for {year}-{month:02d}: {e}")

        attempt += 1

    # 如果失敗或資料仍為空
    print(
        f"Failed to get valid data for {year}-{month:02d} after {max_retries} attempts."
    )
    return pd.DataFrame()


def transform_data(df):
    # filtering and removing outliers
    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    return df


def prepare_data(df):
    return df


def fearure_engineering(df_train, df_val=None):
    categorical = ["PU_DO"]  #'PULocationID', 'DOLocationID']
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    target = "duration"
    y_train = df_train[target].values
    y_val = df_val[target].values

    return X_train, X_val, y_train, y_val, dv


def find_optimal_model(X_train, X_val, y_train, y_val):
    models_folder = Path("models")
    models_folder.mkdir(exist_ok=True)

    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            "learning_rate": 0.09585355369315604,
            "max_depth": 30,
            "min_child_weight": 1.060597050922164,
            "objective": "reg:linear",
            "reg_alpha": 0.018060244040060163,
            "reg_lambda": 0.011658731377413597,
            "seed": 42,
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, "validation")],
            early_stopping_rounds=50,
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

    return params


def train_model(X, Y, params):
    return model


def main():
    df_train = download_data(2021, 1)
    df_train = transform_data(df_train)
    print(f"Downloaded training data: {df_train.shape}")

    df_val = download_data(2021, 2)
    df_val = transform_data(df_val)
    print(f"Downloaded validation data: {df_val.shape}")

    # df_train = prepare_data(df_train)
    X_train, X_val, y_train, y_val, _ = fearure_engineering(df_train, df_val)
    # params = find_optimal_model(X, Y)
    # model = train_model(X, Y, params)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("nyc-taxi-experiment")

    main()
