import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Q1. Downloading the data
df = pd.read_parquet(
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
)
print("Q1. Downloading the data")
print(f"Number of columns: {df.shape[1]}")  # 19


# Q2. Computing duration
df["duration"] = (
    df.tpep_dropoff_datetime - df.tpep_pickup_datetime
).dt.total_seconds() / 60
print("\nQ2. Computing duration")
print(f"Standard deviation of duration: {df['duration'].std():.2f}")  # 42.59


# Q3. Dropping outliers
df_filtered = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
fraction_remaining = df_filtered.shape[0] / df.shape[0]
print("\nQ3. Dropping outliers")
print(f"Fraction of records remaining: {fraction_remaining:.2%}")  # 98.12%


# Q4. One-hot encoding
dv = DictVectorizer()
X_train = dv.fit_transform(
    df_filtered[["PULocationID", "DOLocationID"]].astype(str).to_dict(orient="records")
)
y_train = df_filtered["duration"].values
print("\nQ4. One-hot encoding")
print(
    f"Dimensionality (number of columns) after one-hot encoding: {X_train.shape[1]}"
)  # 515


# Q5. Training a model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
print("\nQ5. Training a model")
print(f"RMSE on train: {rmse_train:.2f}")  # 7.65


# Q6. Evaluating the model
df_val = pd.read_parquet(
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet"
)
df_val["duration"] = (
    df_val.tpep_dropoff_datetime - df_val.tpep_pickup_datetime
).dt.total_seconds() / 60
df_val_filtered = df_val[(df_val["duration"] >= 1) & (df_val["duration"] <= 60)]

X_val = dv.transform(
    df_val_filtered[["PULocationID", "DOLocationID"]]
    .astype(str)
    .to_dict(orient="records")
)
y_val = df_val_filtered["duration"].values

y_pred_val = model.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
print("\nQ6. Evaluating the model")
print(f"RMSE on validation: {rmse_val:.2f}")  # 7.81
