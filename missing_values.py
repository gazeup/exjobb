import pandas as pd
import numpy as np

# Save filepath to variable for easier access
ai41_file_path = '../datasets/ai4i/ai4i2020.csv'

# Read the data and store data in DataFrame titled ai4i_data
ai4i_data = pd.read_csv(ai41_file_path)

# Print number of null values
missing_values_count = ai4i_data.isnull().sum()
print(missing_values_count)
