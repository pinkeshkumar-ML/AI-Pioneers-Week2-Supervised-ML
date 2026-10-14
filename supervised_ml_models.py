"""
Week 2 Task: Supervised Machine Learning Models
Install dependencies: pip install pandas numpy scikit-learn matplotlib
Run: python supervised_ml_models.py
Dataset: student_performance_ml_dataset.csv (synthetic practice data)
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE = Path(__file__).resolve().parent
data = pd.read_csv(BASE / "student_performance_ml_dataset.csv")

# Regression: predict the numeric Final_Score.
X_reg = data[["Study_Hours", "Attendance", "Previous_Score"]]
y_reg = data["Final_Score"]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

regression_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(max_depth=4, random_state=42),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "KNN Regressor": Pipeline([("scale", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=5))]),
}
reg_results = []
for name, model in regression_models.items():
    model.fit(Xr_train, yr_train)
    pred = model.predict(Xr_test)
    reg_results.append({
        "Model": name,
        "MAE": round(mean_absolute_error(yr_test, pred), 3),
        "RMSE": round(mean_squared_error(yr_test, pred) ** 0.5, 3),
        "R2": round(r2_score(yr_test, pred), 3),
    })
pd.DataFrame(reg_results).to_csv(BASE / "regression_results.csv", index=False)
print("\\nREGRESSION RESULTS (lower MAE/RMSE and higher R2 are better)")
print(pd.DataFrame(reg_results).to_string(index=False))

# Classification: predict Pass (1 = pass, 0 = not pass).
X_clf = data[["Study_Hours", "Attendance", "Previous_Score"]]
y_clf = data["Pass"]
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)
classification_models = {
    "Logistic Regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]),
    "Decision Tree Classifier": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN Classifier": Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]),
}
clf_results = []
for name, model in classification_models.items():
    model.fit(Xc_train, yc_train)
    pred = model.predict(Xc_test)
    clf_results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(yc_test, pred), 3),
        "Precision": round(precision_score(yc_test, pred, zero_division=0), 3),
        "Recall": round(recall_score(yc_test, pred, zero_division=0), 3),
        "F1": round(f1_score(yc_test, pred, zero_division=0), 3),
    })
pd.DataFrame(clf_results).to_csv(BASE / "classification_results.csv", index=False)
print("\\nCLASSIFICATION RESULTS (higher values are generally better)")
print(pd.DataFrame(clf_results).to_string(index=False))
print("\\nSaved regression_results.csv and classification_results.csv")
