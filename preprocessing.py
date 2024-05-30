import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load data
def load_data(filepath='../datasets/ai4i/ai4i2020.csv'):
    data = pd.read_csv(filepath)
    return data

# Preprocess features
def preprocess_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Split into training and testdata
def get_train_test_data(data, test_size=0.2, random_state=1):
    y = data['Machine failure']
    X = data[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train_scaled, X_test_scaled = preprocess_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test
