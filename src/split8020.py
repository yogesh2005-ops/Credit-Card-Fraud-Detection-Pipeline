import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load your dataset
df = pd.read_csv("data/creditcard.csv")

# 2. Separate your features (X) from the target answer (y)
X = df.drop(columns=["Class"])
y = df["Class"]

# 3. SPLIT FIRST: Divide into 80% Training data and 20% Testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Create explicit copies to avoid slicing warnings in pandas
X_train = X_train.copy()
X_test = X_test.copy()

# 4. SCALE SECOND: Shrink 'Amount' and 'Time' columns so they use small, uniform scales
scaler = StandardScaler()
cols = ["Amount", "Time"]

# Fit and transform ONLY on the training data
X_train[cols] = scaler.fit_transform(X_train[cols])

# Transform the test data using the training data's scale (No fitting!)
X_test[cols] = scaler.transform(X_test[cols])

# Create complete tables by joining features and answers back together
train_full = X_train.copy()
train_full['Class'] = y_train

test_full = X_test.copy()
test_full['Class'] = y_test

# Save them physically to your data folder
train_full.to_csv("data/creditcard_train.csv", index=False)
test_full.to_csv("data/creditcard_test.csv", index=False)

print("Saved physical CSV files to the data/ directory!")


# === YOUR DAY 2 DELIVERABLE ===
print("=== Day 2 Verification ===")
print(f"Training Fraud Rate: {(y_train.value_counts(normalize=True)[1] * 100):.3f}%")
print(f"Testing Fraud Rate:  {(y_test.value_counts(normalize=True)[1] * 100):.3f}%")
