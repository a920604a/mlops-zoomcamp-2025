import pandas as pd
from sklearn.feature_extraction import DictVectorizer


df = pd.read_parquet(
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
)
# df = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet")

# Q1. Downloading the data
# Read the data for January. How many columns are there?
print("Q1. Downloading the data")
print(df.shape)


# Q2. Computing duration
# What's the standard deviation of the trips duration in January?
df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
print("Q2. Computing duration")
print(df["duration"].std())


# Q3. Dropping outliers
# Next, we need to check the distribution of the duration variable. There are some outliers. Let's remove them and keep only the records where the duration was between 1 and 60 minutes (inclusive).
# What fraction of the records left after you dropped the outliers?
df_drop = df[
    (df["duration"] >= pd.Timedelta(minutes=1))
    & (df["duration"] <= pd.Timedelta(minutes=60))
]
print("Q3. Dropping outliers")
print(f"Fraction of records remaining: {df_drop.shape[0] / df.shape[0]:.2%}")

# Q4. One-hot encoding

print("Q4. One-hot encoding")
# Convert the relevant columns to strings and create a list of dictionaries
data = df_drop[["PULocationID", "DOLocationID"]].astype(str).to_dict(orient="records")

# Initialize the DictVectorizer
dv = DictVectorizer()

# Fit and transform the data to a one-hot encoded feature matrix
feature_matrix = dv.fit_transform(data)

# Get the dimensionality (number of columns)
num_columns = feature_matrix.shape[1]

print(f"Dimensionality (number of columns) after one-hot encoding: {num_columns}")
