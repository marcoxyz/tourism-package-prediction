import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"
df = pd.read_csv(RAW_PATH)
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")
if df["ProdTaken"].isna().any() or set(df["ProdTaken"].unique()) != {0, 1}:
    raise ValueError("ProdTaken must contain both classes 0 and 1 without missing labels.")
print("Dataset registered successfully; the raw CSV is versioned in GitHub.")
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("Class counts:")
print(df["ProdTaken"].value_counts())
