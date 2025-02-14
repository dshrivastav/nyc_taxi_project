import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

# Load preprocessed dataset
df = pd.read_csv("data/processed_taxi_data.csv")

# Define features and target
X = df[['passenger_count', 'trip_distance', 'PULocationID', 'DOLocationID', 'hour', 'day_of_week']]
y = df['trip_duration']

# Split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
model.fit(X_train_scaled, y_train)

# Save trained model & scaler
pickle.dump(model, open("taxi_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Evaluate model
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"✅ Model Training Complete - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
