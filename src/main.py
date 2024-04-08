import pandas

# save filepath to variable for easier access
ai41_file_path = '../datasets/ai4i/ai4i2020.csv'

# Set option to display more columns
pandas.set_option('display.max_columns', None)  # None means unlimited

# read the data and store data in DataFrame titled ai4i_data
ai41_data = pandas.read_csv(ai41_file_path)

# print a summary of the data
print(ai41_data.describe())

