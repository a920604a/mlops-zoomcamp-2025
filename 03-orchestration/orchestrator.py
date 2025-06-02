import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow


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
    return df


def prepare_data(df):
    return df


def fearure_engineering(df):
    return X, Y


def find_optimal_model(X, Y):
    return params


def train_model(X, Y, params):
    return model


def main():
    df_train = download_data(2021, 1)
    print(f"Downloaded training data: {df_train.shape}")
    # df_train = transform_data(df_train)
    # df_train = prepare_data(df_train)
    # X, Y = fearure_engineering(df_train)
    # params = find_optimal_model(X, Y)
    # model = train_model(X, Y, params)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("nyc-taxi-experiment")

    main()
