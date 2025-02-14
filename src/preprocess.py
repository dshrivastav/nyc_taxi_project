import pandas as pd
import numpy as np

# Load dataset from local file
file_path = "yellow_tripdata_2021-07.csv"
df = pd.read_csv(file_path)

# Select useful columns
df = df[['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance', 'PULocationID', 'DOLocationID']]

# Convert datetime columns
df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

# Calculate trip duration in minutes
df['trip_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60

# Remove outliers (trips <1 min or >120 min)
df = df[(df['trip_duration'] > 1) & (df['trip_duration'] < 120)]

# Extract time-based features
df['hour'] = df['pickup_datetime'].dt.hour
df['day_of_week'] = df['pickup_datetime'].dt.weekday

# Drop original datetime columns
df.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], inplace=True)

# Save processed data
df.to_csv("processed_taxi_data.csv", index=False)
print("✅ Data Preprocessing Completed - File Saved: processed_taxi_data.csv")
