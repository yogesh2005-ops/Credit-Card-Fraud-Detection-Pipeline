import pandas as pd

df = pd.read_csv("data/creditcard.csv")
print(df.shape)
counts = df["Class"].value_counts()
percentages = df["Class"].value_counts(normalize=True).mul(100).round(2)

print(pd.DataFrame({"Count": counts, "Percentage": percentages}))
