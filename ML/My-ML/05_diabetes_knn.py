# ML Practical 4: Diabetes Prediction using K-Nearest Neighbors (KNN)
# Dataset: diabetes.csv
#
# Install:
# pip install pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)

# 1. Load dataset
df = pd.read_csv("diabetes.csv")
print("First five records:")
print(df.head())
print("\nDataset shape:", df.shape)

# 2. Data preprocessing
df = df.drop_duplicates()
df = df.dropna()

if "Outcome" not in df.columns:
    raise ValueError("Target column 'Outcome' was not found.")

# In this diabetes dataset, zero is used for missing measurements
# in these columns. Replace those zero values with the column median.
zero_columns = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]
for col in zero_columns:
    if col in df.columns:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

# 3. Separate features and target
X = df.drop(columns=["Outcome"])
y = df["Outcome"].astype(int)

# 4. Split off the final test data
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5. Create validation data to select the best K without using test data
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.20, random_state=42, stratify=y_train_full
)

# 6. Scale using training data only
selection_scaler = StandardScaler()
X_train_scaled = selection_scaler.fit_transform(X_train)
X_val_scaled = selection_scaler.transform(X_val)

# 7. Find K with minimum validation error rate
k_values = range(1, 22, 2)
error_rates = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    val_prediction = model.predict(X_val_scaled)
    error = 1 - accuracy_score(y_val, val_prediction)
    error_rates.append(error)

best_k = list(k_values)[int(np.argmin(error_rates))]
print("\nValidation error rates:")
for k, error in zip(k_values, error_rates):
    print(f"K = {k}: Error Rate = {error * 100:.2f}%")
print("\nBest K value:", best_k)

# 8. Plot K against error rate
plt.figure(figsize=(8, 4))
plt.plot(list(k_values), error_rates, marker="o")
plt.xlabel("K value")
plt.ylabel("Validation error rate")
plt.title("KNN: Finding the Best K")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Train final KNN model on all training data
final_scaler = StandardScaler()
X_train_full_scaled = final_scaler.fit_transform(X_train_full)
X_test_scaled = final_scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_full_scaled, y_train_full)

# 10. Predict test data
y_pred = knn.predict(X_test_scaled)

# 11. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy

print("\n========== FINAL KNN RESULTS ==========")
print("Best K:", best_k)
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Error Rate:", round(error_rate * 100, 2), "%")
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1 Score:", round(f1_score(y_test, y_pred, zero_division=0), 4))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 12. Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Diabetes KNN Confusion Matrix")
plt.tight_layout()
plt.show()

print("\nDiabetes KNN practical completed.")
