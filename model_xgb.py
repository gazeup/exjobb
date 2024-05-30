import pandas as pd
import time
import os
import psutil
from preprocessing import load_data, get_train_test_data
from evaluation import evaluate_algorithms

import xgboost as xgb

def print_process_usage():
    current_process = psutil.Process(os.getpid())
    memory_info = current_process.memory_info()
    memory_usage = memory_info.rss
    print(f"Memory Usage: {memory_usage / (1024 ** 3):.4f} GB")

# Load and preprocess the data
ai4i_data = load_data()
X_train, X_test, y_train, y_test = get_train_test_data(ai4i_data)

# Initialize and train models
model = xgb.XGBClassifier(
    n_estimators=200,            # Number of gradient boosted trees
    max_depth=10,                # Maximum tree depth for base learners
    learning_rate=0.03,          # Boosting learning rate (xgb's "eta")
    subsample=0.7,               # Subsample ratio of the training instance
    colsample_bytree=1.0,        # Subsample ratio of columns when constructing each tree
    use_label_encoder=False,     # Avoids a deprecation warning since XGBoost no longer needs label encoder
    eval_metric='logloss'        # Metric used for evaluation during training
)

# Print usage before training
print_process_usage()

# Start timing for training
#start_training_time = time.perf_counter()

# Fit the model on the training data
model.fit(X_train, y_train)

# End timing for training and calculate the duration
#training_time = time.perf_counter() - start_training_time
#print(f"Training Time: {training_time:.5f} seconds")

# Print usage after training
print_process_usage()

# Start timing for prediction
#start_prediction_time = time.perf_counter()

# Predict on the testing data
predictions = model.predict(X_test)

# End timing for prediction and calculate the duration
#prediction_time = time.perf_counter() - start_prediction_time
#print(f"Prediction Time: {prediction_time:.5f} seconds")


# Calculate the probabilities of predictions
probabilities = model.predict_proba(X_test)[:, 1]

# Calculate and print Accuracy, Precision, Recall, F1 score, ROC and Confusion Matrix
evaluate_algorithms(y_test, predictions, probabilities)