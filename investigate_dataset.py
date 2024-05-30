import pandas as pd
import matplotlib.pyplot as plt

# Save filepath to variable for easier access
ai41_file_path = 'datasets/ai4i/ai4i2020.csv'

# Read the data and store data in DataFrame named ai4i_data
ai4i_data = pd.read_csv(ai41_file_path)

# Print a summary of the data
#print(ai4i_data.describe())

# Print the names of the columns
#print(ai4i_data.columns)

# Value counts for the failure types.
failure_counts = pd.DataFrame({
    'TWF': ai4i_data['TWF'].value_counts(),
    'HDF': ai4i_data['HDF'].value_counts(),
    'PWF': ai4i_data['PWF'].value_counts(),
    'OSF': ai4i_data['OSF'].value_counts(),
    'RNF': ai4i_data['RNF'].value_counts()
})

# Extract and plot the count of '1's for each failure type.
failure_counts = failure_counts.loc[1]

# Count the total number of machine failures
machine_failure_counts = ai4i_data['Machine failure'].value_counts()


# Visualization

# Define custom colors for the charts
custom_colors = ['#ffa600', '#f95d6a', '#68F7AB', '#a05195', '#2f4b7c']

# Pie chart for Machine Failures compared to total number of UIDs
machine_failure_counts.plot(kind='pie', colors=custom_colors,autopct='%1.1f%%')
plt.title('Number of Machine Failures compared to the total number of UID')
plt.ylabel('')
plt.show()

# Bar chart to show failure types
plt.figure(figsize=(10, 6))
# Create the bar plot
bars = plt.bar(failure_counts.index, failure_counts.values, color=custom_colors)

# Adding the counts above the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center')

plt.title('Number of Failures by Type')
plt.xlabel('Type of Failure')
plt.ylabel('Counts')
plt.xticks(rotation=0)
plt.show()