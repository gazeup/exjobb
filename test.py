import pandas as pd

# save filepath to variable for easier access
ai41_file_path = '../datasets/ai4i/ai4i2020.csv'

# Set option to display more columns
pd.set_option('display.max_columns', None)  # None means unlimited

# read the data and store data in DataFrame titled ai4i_data
ai4i_data = pd.read_csv(ai41_file_path)



# PRINTING DATA

# print a summary of the data
#print(ai4i_data.describe())

# Print the names of the columns
#print(ai4i_data.columns)




# HANDLE MISSING VALUES

# Drop missing values
ai4i_data = ai4i_data.dropna(axis=0)




# PICK A PREDICTION TARGET

y = ai4i_data['Tool wear [min]']




# PICK FEATURES

ai4i_features = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
]

X = ai4i_data[ai4i_features]





# QUICK REVIEW OF THE FEATURES DATA
#print(X.describe())

#print(X.head())



# BUILDING MODEL
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
ai4i_model = DecisionTreeRegressor(random_state=1)

# Fit model
ai4i_model.fit(X, y)


# TEST TO PREDICT
predictions = ai4i_model.predict(X)
print(predictions)



# SUMMARIZE THE MODEL QUALITY INTO ONE VALUE

from sklearn.metrics import mean_absolute_error

#predicted_tool_wear = ai4i_model.predict(X)
#print(mean_absolute_error(y, predicted_tool_wear))