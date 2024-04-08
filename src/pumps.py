import pandas as pd

# save filepath to variable for easier access
pumps_file_path = '../datasets/pump_sensor_data/sensor.csv'

# Set option to display more columns
pd.set_option('display.max_columns', None)  # None means unlimited

# read the data and store data in DataFrame
pumps_data = pd.read_csv(pumps_file_path)

# print a summary of the data
print(pumps_data.describe())