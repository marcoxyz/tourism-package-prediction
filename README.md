# Tourism Package Prediction — Full Code
Authorized platform: GitHub Actions + Streamlit Community Cloud.

Pipeline: data registration -> stratified train/test preparation -> XGBoost
training with GridSearchCV and MLflow -> model committed to main.

Deploy from main with entry point tourism_project/deployment/app.py.
Use the same Python version as .github/workflows/pipeline.yml.
Dependencies: tourism_project/deployment/requirements.txt.
The fixed classification threshold is 0.45 in both evaluation and the app.
Sales-interaction fields are excluded for pre-contact prediction.
Review the actual reports and workflow artifacts; historical purchases do not
directly establish responses to the new Wellness package.

Structure:
- .github/workflows/pipeline.yml
- tourism_project/data/tourism.csv
- tourism_project/model_building/data_register.py, prep.py, train.py
- tourism_project/deployment/app.py, requirements.txt, Dockerfile, trained model
- tourism_project/hosting/hosting.py
- tourism_project/reports/ (generated metrics and CV results)
- tourism_project/requirements.txt
