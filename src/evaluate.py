import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report

print("=== Day 4: Threshold Optimization ===")

# 1. Load the physical datasets
train_df = pd.read_csv("data/creditcard_train.csv")
test_df = pd.read_csv("data/creditcard_test.csv")

X_train = train_df.drop(columns=["Class"])
y_train = train_df["Class"]
X_test = test_df.drop(columns=["Class"])
y_test = test_df["Class"]

# 2. Retrain the winning Random Forest model
print("Training the winning Random Forest model...")
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

# 3. Get raw probability scores for the test set
proba = rf.predict_proba(X_test)[:, 1]

# 4. Calculate precision and recall across all potential thresholds
precision, recall, thresholds = precision_recall_curve(y_test, proba)

# 5. Business rule: Find a threshold targeting at least 80% recall with best precision
target_recall = 0.80
valid_indices = np.where(recall[:-1] >= target_recall)[0]
best_idx = valid_indices[np.argmax(precision[valid_indices])]
chosen_threshold = thresholds[best_idx]

# 6. Apply our custom threshold to make final classifications
y_pred = (proba >= chosen_threshold).astype(int)

print(f"\n[+] Optimal Selected Risk Threshold: {chosen_threshold:.4f}")
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"-> Legitimate Transactions Correctly Passed (True Negatives): {cm[0][0]}")
print(f"-> Legitimate Transactions Falsely Blocked (False Positives): {cm[0][1]} <--- Customer Friction")
print(f"-> Fraud Cases Missed (False Negatives): {cm[1][0]} <--- Bank Loss")
print(f"-> Fraud Cases Successfully Caught (True Positives): {cm[1][1]}")

print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred, digits=4))
