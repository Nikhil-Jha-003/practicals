ML PRACTICAL MANUAL - CORRECTED PYTHON CODE

This package contains the five practicals listed in the ML manual.

PRACTICAL 1 - Uber ride fare prediction
File: 01_uber_fare_prediction.py
CSV: uber.csv
- Preprocessing, outlier handling, Haversine trip distance and time features
- Linear Regression and Random Forest Regression
- MAE, MSE, RMSE and R2 score comparison

PRACTICAL 2 - Email spam detection
File: 02_email_spam_knn_svm.py
CSV: emails.csv
- Uses the supplied preprocessed numeric email-word features
- KNN and SVM classifiers
- Accuracy, confusion matrix and classification report

PRACTICAL 3 - Gradient Descent
File: 03_gradient_descent.py
- Minimizes y = (x + 3)^2 starting from x = 2
- Prints iterations and plots the descent path
- The manual's worked example may show a different expression/start value;
  follow the exact expression and starting value your instructor specifies.

PRACTICAL 4 - Bank customer churn using Neural Network
File: 04_bank_customer_churn_neural_network.py
CSV: Churn_Modelling.csv
- Preprocessing, encoding, train/test split and normalization
- Neural network, accuracy, confusion matrix and classification report

PRACTICAL 5 - KNN on Diabetes dataset
File: 05_diabetes_knn.py
CSV: diabetes.csv
- Replaces invalid zero measurements with median values
- Selects K using validation error rate
- Accuracy, error rate, confusion matrix, precision, recall and F1 score

HOW TO RUN
1. Extract this ZIP file.
2. Put the four CSV files in the same folder as the Python scripts.
3. Install dependencies:
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
4. Run the required practical:
   python 01_uber_fare_prediction.py
   python 02_email_spam_knn_svm.py
   python 03_gradient_descent.py
   python 04_bank_customer_churn_neural_network.py
   python 05_diabetes_knn.py

CSV FILENAMES
- uber.csv
- emails.csv
- Churn_Modelling.csv
- diabetes.csv

If your file names differ, either rename the CSV files or update the
corresponding pd.read_csv(...) line in the script.
