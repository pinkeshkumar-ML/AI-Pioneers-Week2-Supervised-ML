# Week 2: Supervised Machine Learning Models

## Project
Student Performance Prediction with Regression and Classification

## Files
- `student_performance_ml_dataset.csv`: synthetic practice dataset
- `supervised_ml_models.py`: trains and evaluates regression and classification models
- `regression_results.csv`: MAE, RMSE, and R² comparison
- `classification_results.csv`: accuracy, precision, recall, and F1 comparison
- `Week2_Supervised_ML_Report.docx`: task report

## Setup
Install Python 3 and dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib
```

Run:
```bash
python supervised_ml_models.py
```

The script trains Linear Regression, Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors models. It uses a reproducible 80/20 train/test split. The dataset is synthetic and intended for learning; results should not be interpreted as real student-performance findings.
