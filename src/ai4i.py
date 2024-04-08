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
# print(ai4i_data.columns)




# Handle missing values

# Drop missing values
ai4i_data = ai4i_data.dropna(axis=0)

