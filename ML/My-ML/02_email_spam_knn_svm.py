# ML Practical 2: Email Spam Detection using KNN and SVM
# Dataset: emails.csv (preprocessed numeric email-word features)
#
# Install:
# pip install pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
df = pd.read_csv("emails.csv", encoding="latin-1")
print("First five records:")
print(df.head())
print("\nDataset shape:", df.shape)

# 2. Check target column
if "Prediction" not in df.columns:
    raise ValueError(
        "Target column 'Prediction' was not found in emails.csv."
    )

# 3. Separate target and input features
y = pd.to_numeric(df["Prediction"], errors="coerce")
X = df.drop(columns=["Prediction", "Email No."], errors="ignore")

# Convert all input columns to numeric. Invalid values become missing.
X = X.apply(pd.to_numeric, errors="coerce")

# Remove rows with missing target values
valid_rows = y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows].astype(int)

# Confirm binary labels
if not set(y.unique()).issubset({0, 1}):
    raise ValueError(
        f"Expected Prediction labels 0 and 1, found {sorted(y.unique())}"
    )

print("\nClass distribution (0 = Not Spam, 1 = Spam):")
print(y.value_counts())

# 4. Split dataset using stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5. Build KNN model. Impute and scale within the pipeline to avoid leakage.
# The supplied emails.csv has about 3,000 word-feature columns.
# Select the 100 most informative non-negative word features to make KNN
# faster while fitting feature selection only on training data.
number_of_features = min(100, X_train.shape[1])

knn_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("feature_selection", SelectKBest(score_func=chi2, k=number_of_features)),
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=5, algorithm="brute"))
])

knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

print("\n========== KNN RESULTS ==========")
print("Accuracy:", round(accuracy_score(y_test, knn_pred) * 100, 2), "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
print("Classification Report:\n",
      classification_report(y_test, knn_pred, zero_division=0))

# 6. Build SVM model
svm_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("feature_selection", SelectKBest(score_func=chi2, k=number_of_features)),
    ("scaler", StandardScaler()),
    ("classifier", LinearSVC(random_state=42, max_iter=10000))
])

svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

print("\n========== SVM RESULTS ==========")
print("Accuracy:", round(accuracy_score(y_test, svm_pred) * 100, 2), "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))
print("Classification Report:\n",
      classification_report(y_test, svm_pred, zero_division=0))

# 7. Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, prediction, title in [
    (axes[0], knn_pred, "KNN Confusion Matrix"),
    (axes[1], svm_pred, "SVM Confusion Matrix")
]:
    sns.heatmap(
        confusion_matrix(y_test, prediction),
        annot=True, fmt="d", cmap="Blues", ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()

# 8. Compare model accuracy
results = pd.DataFrame({
    "Model": ["KNN", "SVM"],
    "Accuracy (%)": [
        accuracy_score(y_test, knn_pred) * 100,
        accuracy_score(y_test, svm_pred) * 100
    ]
})
print("\nModel comparison:")
print(results)

print("\nEmail spam classification practical completed.")
