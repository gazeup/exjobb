import pandas as pd
from preprocessing import load_data, get_train_test_data
from evaluation import evaluate_algorithms
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


# Load and preprocess the data
ai4i_data = load_data()
X_train, X_test, y_train, y_test = get_train_test_data(ai4i_data)


# Initialize the Decision Tree Classifier
model = DecisionTreeClassifier()


# Define the extended parameter grid for the Decision Tree
param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'min_samples_split': [11, 12, 13, 14],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7],
    'max_features': [None, 'auto', 'sqrt', 'log2']
}


# Setup Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')


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