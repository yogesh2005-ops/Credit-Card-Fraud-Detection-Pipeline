import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

print("=== Day 3: Training Competitor Models ===")

# 1. Read directly from your newly created training/testing CSV files
print("Loading physical split datasets...")
train_df = pd.read_csv("data/creditcard_train.csv")
test_df = pd.read_csv("data/creditcard_test.csv")

# 2. Separate features (X) from the target answer (y) for both sets
X_train = train_df.drop(columns=["Class"])
y_train = train_df["Class"]

X_test = test_df.drop(columns=["Class"])
y_test = test_df["Class"]

# 3. Define your model competitors with balanced weights
models = {
    "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,            # Built with 100 trees to balance speed and power
        class_weight="balanced_subsample",
        n_jobs=-1,                   # Uses all CPU cores to compute faster
        random_state=42,
    ),
}

global_results = {}

# 4. Train each model and calculate their performance scores
for name, model in models.items():
    print(f"\nTraining {name} on creditcard_train.csv...")
    model.fit(X_train, y_train)
    
    # Extract prediction probability scores for the fraud class (Class 1)
    proba = model.predict_proba(X_test)[:, 1]
    global_results[name] = proba
    
    # Calculate PR-AUC score over the unseen test set
    pr_auc = average_precision_score(y_test, proba)
    print(f"-> {name} PR-AUC Score: {pr_auc:.4f}")

print("\n=== Day 3 Complete ===")
