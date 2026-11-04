# ML Practical 1: Uber Ride Fare Prediction
# Models: Linear Regression and Random Forest Regression
# Dataset: uber.csv
#
# Install required packages:
# pip install pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("uber.csv")
print("First five records:")
print(df.head())
print("\nDataset shape:", df.shape)

# 2. Remove unnecessary columns and duplicate/missing records
df = df.drop(columns=["Unnamed: 0", "key"], errors="ignore")
df = df.drop_duplicates()

required = [
    "fare_amount", "pickup_datetime", "pickup_longitude",
    "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
    "passenger_count"
]
missing_columns = [c for c in required if c not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df = df.dropna(subset=required).copy()

# 3. Convert data types
df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"], errors="coerce", utc=True
)
numeric_cols = [
    "fare_amount", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "passenger_count"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["pickup_datetime"] + numeric_cols)

# 4. Remove clearly invalid fare, passenger and coordinate values
df = df[
    (df["fare_amount"] > 0) &
    (df["fare_amount"] <= 250) &
    (df["passenger_count"].between(1, 6)) &
    (df["pickup_latitude"].between(-90, 90)) &
    (df["dropoff_latitude"].between(-90, 90)) &
    (df["pickup_longitude"].between(-180, 180)) &
    (df["dropoff_longitude"].between(-180, 180))
].copy()

# 5. Identify and remove fare outliers using IQR
Q1 = df["fare_amount"].quantile(0.25)
Q3 = df["fare_amount"].quantile(0.75)
IQR = Q3 - Q1
lower_limit = max(0, Q1 - 1.5 * IQR)
upper_limit = Q3 + 1.5 * IQR

outlier_count = (
    (df["fare_amount"] < lower_limit) |
    (df["fare_amount"] > upper_limit)
).sum()
print("\nFare outliers detected:", outlier_count)

df = df[df["fare_amount"].between(lower_limit, upper_limit)].copy()

# 6. Extract time-based features
df["hour"] = df["pickup_datetime"].dt.hour
df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
df["month"] = df["pickup_datetime"].dt.month

# 7. Calculate trip distance using the Haversine formula
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return 6371 * 2 * np.arcsin(np.sqrt(a))  # distance in kilometres

df["distance_km"] = haversine_distance(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"]
)

# Remove zero-distance and unusually long trips
df = df[(df["distance_km"] > 0) & (df["distance_km"] <= 100)].copy()

print("\nCleaned dataset shape:", df.shape)
print("\nMissing values after preprocessing:")
print(df.isnull().sum())

# 8. Visualize fare outliers and correlation
plt.figure(figsize=(8, 4))
sns.boxplot(x=df["fare_amount"])
plt.title("Uber Fare Amount - Outlier Detection")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
sns.heatmap(
    df[["fare_amount", "distance_km", "passenger_count",
        "hour", "day_of_week", "month"]].corr(),
    annot=True, cmap="coolwarm", fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 9. Select input and target variables
X = df[[
    "distance_km", "passenger_count", "hour", "day_of_week", "month"
]]
y = df["fare_amount"]

# 10. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 11. Train Linear Regression (scaling is fitted only on training data)
linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

# 12. Train Random Forest Regression
rf_model = RandomForestRegressor(
    n_estimators=100, random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# 13. Evaluate both models
def evaluate_model(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    print(f"\n{name}")
    print("MAE :", round(mae, 3))
    print("MSE :", round(mse, 3))
    print("RMSE:", round(rmse, 3))
    print("R2 Score:", round(r2, 4))
    return [mae, mse, rmse, r2]

linear_scores = evaluate_model("Linear Regression Results", y_test, linear_pred)
rf_scores = evaluate_model("Random Forest Results", y_test, rf_pred)

comparison = pd.DataFrame(
    [linear_scores, rf_scores],
    columns=["MAE", "MSE", "RMSE", "R2 Score"],
    index=["Linear Regression", "Random Forest"]
)
print("\nModel Comparison:")
print(comparison)

# 14. Predict fare for a sample ride
sample_ride = pd.DataFrame({
    "distance_km": [5.0],
    "passenger_count": [2],
    "hour": [14],
    "day_of_week": [2],
    "month": [6]
})
print("\nPredicted fare using Linear Regression:",
      round(linear_model.predict(sample_ride)[0], 2))
print("Predicted fare using Random Forest:",
      round(rf_model.predict(sample_ride)[0], 2))

print("\nUber fare prediction practical completed.")
