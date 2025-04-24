import pandas as pd


df = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet")
# df = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet")

# Q1. Downloading the data 
# Read the data for January. How many columns are there?
print(df.shape)  
# Q2. Computing duration
# What's the standard deviation of the trips duration in January?
df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime

print(df['duration'].std())


# Q3. Dropping outliers
# Next, we need to check the distribution of the duration variable. There are some outliers. Let's remove them and keep only the records where the duration was between 1 and 60 minutes (inclusive).
# What fraction of the records left after you dropped the outliers?
df_drop = df[(df['duration'] >= pd.Timedelta(minutes=1)) & (df['duration'] <= pd.Timedelta(minutes=60))]
df_drop.shape[0]/df.shape[0]
