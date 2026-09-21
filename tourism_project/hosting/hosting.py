import os
import pandas as pd
import joblib

folder = "tourism_project/deployment"
for name in ["app.py", "requirements.txt", "Dockerfile", "best_tourism_package_model_v1.joblib"]:
    if not os.path.isfile(os.path.join(folder, name)):
        raise FileNotFoundError(f"Missing deployment file: {name}")
model = joblib.load(os.path.join(folder, "best_tourism_package_model_v1.joblib"))
sample = pd.read_csv("Xtest.csv").head(1)
probability = float(model.predict_proba(sample)[0, 1])
assert 0 <= probability <= 1
print("Model loaded and prediction succeeded.")
print("Streamlit entry point: tourism_project/deployment/app.py")
print("Connect the GitHub repository and main branch in Streamlit Community Cloud.")
