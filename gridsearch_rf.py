import pandas as pd
from preprocessing import load_data, get_train_test_data
from evaluation import evaluate_algorithms
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# Load and preprocess the data
ai4i_data = load_data()
X_train, X_test, y_train, y_test = get_train_test_data(ai4i_data)


# Initialize the RandomForest model
model = RandomForestClassifier()


# Define the parameter grid for the RandomForest
param_grid = {
    'n_estimators': [200, 300, 400],  # Number of trees in the forest
    'max_depth': [None, 11, 12, 13, 14],  # Maximum depth of each tree
    'min_samples_split': [ 3, 4, 5],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2],  # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2']  # Number of features to consider when looking for the best split
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