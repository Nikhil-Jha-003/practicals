# ML Practical 3: Bank Customer Churn Prediction using Neural Network
# Dataset: Churn_Modelling.csv
#
# Install:
# pip install pandas numpy matplotlib seaborn scikit-learn tensorflow

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load dataset
df = pd.read_csv("Churn_Modelling.csv")
print("First five records:")
print(df.head())
print("\nDataset shape:", df.shape)

# 2. Data preprocessing
df = df.drop_duplicates()
df = df.dropna()

# Remove identifiers; they are not useful predictive features.
df = df.drop(
    columns=["RowNumber", "CustomerId", "Surname"],
    errors="ignore"
)

if "Exited" not in df.columns:
    raise ValueError("Target column 'Exited' was not found.")

# 3. Separate features and target
X = df.drop(columns=["Exited"])
y = df["Exited"].astype(int)

# Convert categorical variables into numeric dummy variables.
X = pd.get_dummies(X, columns=["Geography", "Gender"],
                   drop_first=True, dtype=int)

# 4. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5. Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype("float32")
X_test = scaler.transform(X_test).astype("float32")

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# 6. Build neural network
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(16, activation="relu"),
    Dropout(0.2),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

# 7. Compile neural network
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 8. Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# 9. Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.20,
    callbacks=[early_stop],
    verbose=1
)

# 10. Predict test data
probabilities = model.predict(X_test, verbose=0).ravel()
y_pred = (probabilities >= 0.5).astype(int)

# 11. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 12. Display confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Bank Customer Churn - Confusion Matrix")
plt.tight_layout()
plt.show()

# 13. Plot training and validation accuracy
plt.figure(figsize=(8, 4))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Neural Network Model Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

print("\nBank customer churn prediction practical completed.")
