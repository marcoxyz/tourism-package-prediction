import os
import streamlit as st
import pandas as pd
import joblib

model_path = os.path.join(os.path.dirname(__file__), "best_tourism_package_model_v1.joblib")
if not os.path.exists(model_path):
    st.error("Wait for a successful GitHub Actions training run before deploying the app.")
    st.stop()
model = joblib.load(model_path)

st.title("Tourism Package Prediction")
st.write("Enter customer information already available before sales outreach.")
Age = st.slider("Age", 18, 100, 35)
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.slider("Total travelers, including children", 1, 10, 3)
PreferredPropertyStar = st.selectbox("Preferred hotel stars", [3, 4, 5])
MaritalStatus = st.selectbox("Marital status", ["Married", "Single", "Divorced", "Unmarried"])
NumberOfTrips = st.slider("Trips per year", 0, 30, 3)
Passport = st.selectbox("Valid passport?", ["Yes", "No"])
OwnCar = st.selectbox("Owns a car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.slider("Children under age 5", 0, 9, 1)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly income (dataset currency units)", min_value=0.0, value=23000.0)

input_data = pd.DataFrame([{
    "Age": Age, "CityTier": CityTier, "Occupation": Occupation, "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar, "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips, "Passport": int(Passport == "Yes"),
    "OwnCar": int(OwnCar == "Yes"), "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation, "MonthlyIncome": MonthlyIncome,
}])
classification_threshold = 0.45 
if st.button("Predict"):
    if NumberOfChildrenVisiting >= NumberOfPersonVisiting:
        st.error("Total travelers must include at least one adult plus the children.")
    else:
        probability = float(model.predict_proba(input_data)[0, 1])
        prediction = int(probability >= classification_threshold)
        st.write(f"Model purchase score: {probability:.1%}")
        if prediction:
            st.success("Customer is classified as a potential purchaser.")
        else:
            st.info("Customer is classified as a non-purchaser at this threshold.")
        st.caption("Threshold: 45%. This is a model score, not a guaranteed or calibrated purchase probability.")
