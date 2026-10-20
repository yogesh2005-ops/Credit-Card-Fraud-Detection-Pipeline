import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve

# Load data and train model exactly like Day 4
train_df = pd.read_csv("data/creditcard_train.csv")
test_df = pd.read_csv("data/creditcard_test.csv")

X_train, y_train = train_df.drop(columns=["Class"]), train_df["Class"]
X_test, y_test = test_df.drop(columns=["Class"]), test_df["Class"]

rf = RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
proba = rf.predict_proba(X_test)[:, 1]

# Calculate curve values
precision, recall, thresholds = precision_recall_curve(y_test, proba)

# Generate Plot
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label='Random Forest (PR-AUC = 0.8597)', color='navy', lw=2)
plt.axvline(x=0.8367, color='red', linestyle='--', label='Optimized Threshold Target (0.3000)')
plt.xlabel('Recall (Fraud Catch Rate)')
plt.ylabel('Precision (Alarm Trustworthiness)')
plt.title('Precision-Recall Curve & Chosen Operational Threshold')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.savefig("pr_curve.png", dpi=300, bbox_inches='tight')
print("Saved pr_curve.png successfully!")
