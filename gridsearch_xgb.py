import pandas as pd
from preprocessing import load_data, get_train_test_data
from evaluation import evaluate_algorithms
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


# Load and preprocess the data
ai4i_data = load_data()
X_train, X_test, y_train, y_test = get_train_test_data(ai4i_data)


# Initialize the XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')


# Define the parameter grid for XGBoost
param_grid = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [10, 11],
    'learning_rate': [0.01, 0.015, 0.02, 0.025, 0.03],
    'subsample': [0.6, 0.65, 0.7, 0.75],
    'colsample_bytree': [1.0]
}


# Setup Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)


# Perform grid search to find the best parameters
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)


# Use the best model found by GridSearchCV
best_model = grid_search.best_estimator_


# Predict on the testing data using the best model
predictions = best_model.predict(X_test)


# Calculate the probabilities of predictions for the positive class
probabilities = best_model.predict_proba(X_test)[:, 1]


# Calculate and print Accuracy, Precision, Recall, F1 score, ROC and Confusion Matrix
evaluate_algorithms(y_test, predictions, probabilities)