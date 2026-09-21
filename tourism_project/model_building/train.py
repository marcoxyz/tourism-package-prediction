import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tourism-training-experiment")
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv")["ProdTaken"]
ytest = pd.read_csv("ytest.csv")["ProdTaken"]

numeric_features = [
    "Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = ["Occupation", "Gender", "MaritalStatus", "Designation"]
class_weight = float(ytrain.value_counts()[0] / ytrain.value_counts()[1])
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, n_jobs=1)
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}
model_pipeline = make_pipeline(preprocessor, xgb_model)
os.makedirs("tourism_project/deployment", exist_ok=True)
os.makedirs("tourism_project/reports", exist_ok=True)

with mlflow.start_run(run_name="tourism-grid-search"):
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="recall",
                               n_jobs=2, error_score="raise")
    grid_search.fit(Xtrain, ytrain)
    results = grid_search.cv_results_
    for i, params in enumerate(results["params"]):
        with mlflow.start_run(run_name=f"candidate-{i + 1}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_recall", float(results["mean_test_score"][i]))
            mlflow.log_metric("std_cv_recall", float(results["std_test_score"][i]))
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_params({"scale_pos_weight": class_weight, "random_state": 42,
                       "cv_folds": 5, "classification_threshold": 0.45})
    best_model = grid_search.best_estimator_
    print("Best parameters:", grid_search.best_params_)
    print("Best CV recall (estimator default threshold 0.50):", grid_search.best_score_)

    classification_threshold = 0.45
    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= classification_threshold).astype(int)
    y_pred_test = (best_model.predict_proba(Xtest)[:, 1] >= classification_threshold).astype(int)
    train_report = classification_report(ytrain, y_pred_train, output_dict=True, zero_division=0)
    test_report = classification_report(ytest, y_pred_test, output_dict=True, zero_division=0)
    metrics = {
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1": test_report["1"]["f1-score"],
    }
    mlflow.log_metrics(metrics)
    print("TRAIN classification report:")
    print(classification_report(ytrain, y_pred_train, zero_division=0))
    print("TEST classification report:")
    print(classification_report(ytest, y_pred_test, zero_division=0))
    matrix = confusion_matrix(ytest, y_pred_test, labels=[0, 1])
    print("Confusion matrix [TN FP; FN TP]:")
    print(matrix)

    pd.DataFrame([metrics]).to_csv("tourism_project/reports/metrics.csv", index=False)
    pd.DataFrame(results).to_csv("tourism_project/reports/cv_results.csv", index=False)
    pd.DataFrame(matrix, index=["actual_0", "actual_1"],
                 columns=["predicted_0", "predicted_1"]).to_csv("tourism_project/reports/confusion_matrix.csv")
    model_path = "tourism_project/deployment/best_tourism_package_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    mlflow.log_artifacts("tourism_project/reports", artifact_path="reports")
    print("Model saved:", model_path)
