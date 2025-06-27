# task4.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 1. Load the dataset
data = pd.read_csv('data.csv')

# Assume the last column is the target
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# Convert categorical columns to numeric using one-hot encoding
X = pd.get_dummies(X)

# Convert y to numeric if it's categorical
y = y.map({'B': 0, 'M': 1})

# Drop columns that are completely NaN (like Unnamed: 32)
X = X.dropna(axis=1, how='all')

# Drop any rows with NaN in features
X = X.dropna()
y = y.loc[X.index]  # Ensure y matches X's index

# Remove rows where y is NaN
mask = ~y.isna()
X = X[mask]
y = y[mask]

# Debugging: Check data after cleaning
print("Number of samples after cleaning:", len(X))
print("Number of NaN in y:", y.isna().sum())
print("First few rows of X:\n", X.head())
print("First few values of y:\n", y.head())

# 2. Train/test split and standardize features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Fit a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 4. Evaluate with confusion matrix, precision, recall, ROC-AUC
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("Confusion Matrix:\n", cm)
print("Precision:", precision)
print("Recall:", recall)
print("ROC-AUC:", roc_auc)

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# 5. Tune threshold and explain sigmoid function
# Default threshold is 0.5, let's try 0.3
threshold = 0.3
y_pred_new = (y_proba >= threshold).astype(int)
cm_new = confusion_matrix(y_test, y_pred_new)
precision_new = precision_score(y_test, y_pred_new)
recall_new = recall_score(y_test, y_pred_new)

print(f"\nWith threshold = {threshold}:")
print("Confusion Matrix:\n", cm_new)
print("Precision:", precision_new)
print("Recall:", recall_new)

# Sigmoid explanation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print("\nThe sigmoid function maps any value to a probability between 0 and 1, which is used in logistic regression to predict binary outcomes.")
