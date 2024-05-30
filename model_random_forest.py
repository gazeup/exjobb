import pandas as pd
import time
import os
import psutil
from preprocessing import load_data, get_train_test_data
from evaluation import evaluate_algorithms

from sklearn.ensemble import RandomForestClassifier

def print_process_usage():
    current_process = psutil.Process(os.getpid())
    memory_info = current_process.memory_info()
    memory_usage = memory_info.rss
    print(f"Memory Usage: {memory_usage / (1024 ** 3):.4f} GB")

# Load and preprocess the data
ai4i_data = load_data()
X_train, X_test, y_train, y_test = get_train_test_data(ai4i_data)

# Initialize and train models
model = RandomForestClassifier(
    n_estimators=200,     # Number of trees in the forest
    max_depth=None,       # Maximum depth of each tree, 'None' means no limit
    min_samples_split=3,  # Minimum number of samples required to split an internal node
    min_samples_leaf=1,   # Minimum number of samples required to be at a leaf node
    max_features='sqrt'   # Number of features to consider when looking for the best split: 'sqrt' is typically sqrt(n_features)
)

# Print usage before training
#print_process_usage()

# Start timing for training
start_training_time = time.perf_counter()

# Fit the model on the training data
model.fit(X_train, y_train)

# End timing for training and calculate the duration
training_time = time.perf_counter() - start_training_time
print(f"Training Time: {training_time:.5f} seconds")


# Print usage after training
#print_process_usage()

# Start timing for prediction
start_prediction_time = time.perf_counter()

# Predict on the testing data
predictions = model.predict(X_test)

# End timing for prediction and calculate the duration
prediction_time = time.perf_counter() - start_prediction_time
print(f"Prediction Time: {prediction_time:.5f} seconds")


# Calculate the probabilities of predictions
probabilities = model.predict_proba(X_test)[:, 1]

# Calculate and print Accuracy, Precision, Recall, F1 score, ROC and Confusion Matrix
evaluate_algorithms(y_test, predictions, probabilities)