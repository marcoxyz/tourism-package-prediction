import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")
index_columns = [c for c in df.columns if c.startswith("Unnamed:")]
df = df.drop(columns=index_columns).drop_duplicates().copy()
if df["CustomerID"].isna().any() or df["CustomerID"].duplicated().any():
    raise ValueError("Missing or repeated customer IDs require review before splitting.")
for col in df.select_dtypes(include="object"):
    df[col] = df[col].str.strip().replace("", float("nan"))
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# Remove identifiers and sales-interaction information unavailable before outreach.
df = df.drop(columns=["CustomerID", "TypeofContact", "DurationOfPitch",
                      "NumberOfFollowups", "ProductPitched", "PitchSatisfactionScore"])
if df.isna().any().any():
    raise ValueError("Missing values detected. Add training-only imputation before proceeding.")
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"].astype(int)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("Clean rows:", len(df))
print("Removed index columns:", index_columns)
print("Train shape:", Xtrain.shape, "Test shape:", Xtest.shape)
print("Training target distribution:")
print(ytrain.value_counts(normalize=True))
print("Test target distribution:")
print(ytest.value_counts(normalize=True))
print("Categorical values remain strings; preprocessing is fitted during training.")
