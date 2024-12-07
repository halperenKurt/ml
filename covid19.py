import pandas as pd
import joblib
import os

# File paths for joblib datasets
train_joblib_path = 'final_data.joblib'
test_joblib_path = 'test_data.joblib'

# Check if joblib files exist
if os.path.exists(train_joblib_path) and os.path.exists(test_joblib_path):
    # Load the joblib files
    data = joblib.load(train_joblib_path)
    test_data = joblib.load(test_joblib_path)
    print("Joblib files loaded.")
else:
    # Paths for raw CSV files
    file_paths_for_train = [
        'datasets/01-01-2021.csv',
        'datasets/01-02-2021.csv',
        'datasets/01-03-2021.csv',
        'datasets/01-04-2021.csv',
        'datasets/01-05-2021.csv',
        'datasets/01-06-2021.csv',
        'datasets/01-07-2021.csv',
        'datasets/01-08-2021.csv',
        'datasets/01-09-2021.csv',
        'datasets/01-10-2021.csv',
        'datasets/01-11-2021.csv',
        'datasets/01-12-2021.csv',
        'datasets/01-13-2021.csv',
        'datasets/01-14-2021.csv',
        'datasets/01-15-2021.csv',
        'datasets/01-16-2021.csv',
        'datasets/01-17-2021.csv',
        'datasets/01-18-2021.csv',
        'datasets/01-19-2021.csv',
        'datasets/01-20-2021.csv',
        'datasets/01-21-2021.csv',
        'datasets/01-22-2021.csv',
        'datasets/01-23-2021.csv',
        'datasets/01-24-2021.csv',
        'datasets/01-25-2021.csv',
        'datasets/01-26-2021.csv',
        'datasets/01-27-2021.csv',
        'datasets/01-28-2021.csv',
        'datasets/01-29-2021.csv',
        'datasets/01-30-2021.csv',
    ]
    file_paths_for_test = [
        'datasets/02-01-2021.csv',
        'datasets/02-02-2021.csv',
        'datasets/02-03-2021.csv',
        'datasets/02-04-2021.csv',
        'datasets/02-05-2021.csv',
    ]

    # Read and concatenate all train files
    dataframes = [pd.read_csv(file_path) for file_path in file_paths_for_train]
    data = pd.concat(dataframes, ignore_index=True)

    # Read and concatenate all test files
    test_dataframes = [pd.read_csv(file_path) for file_path in file_paths_for_test]
    test_data = pd.concat(test_dataframes, ignore_index=True)

    # Save the concatenated data to joblib files
    joblib.dump(data, train_joblib_path)
    joblib.dump(test_data, test_joblib_path)
    print("Joblib files created and saved.")

# Begin data processing
print("Train dataset shape:", data.shape)
print("Test dataset shape:", test_data.shape)

# Count and percentage of zero values in the "Active" column
active_zeros_count = (data['Active'] == 0).sum()
active_zeros_percentage = (active_zeros_count / len(data)) * 100

print("Number of zeros in the 'Active' column:", active_zeros_count)
print("Percentage of zeros in the 'Active' column:", active_zeros_percentage)

# Filter rows where "Active" is zero
zero_active_cases = data[data['Active'] == 0]

# Summary of "Confirmed," "Deaths," and "Recovered" columns
relation_summary = zero_active_cases[['Confirmed', 'Deaths', 'Recovered']].describe()
print("Summary of 'Confirmed,' 'Deaths,' and 'Recovered' columns (0 active cases):")
print(relation_summary)

# Identify inconsistencies in the "Active" column
calculated_active = data['Confirmed'] - (data['Deaths'] + data['Recovered'])
inconsistencies = data[data['Active'] != calculated_active]
print(f"Number of inconsistent values: {len(inconsistencies)}")

# Correct inconsistent values
data['Active'] = calculated_active

# Check for negative values in numeric columns
numeric_columns = ['Confirmed', 'Deaths', 'Recovered', 'Active']
for col in numeric_columns:
    negative_count = (data[col] < 0).sum()
    if negative_count > 0:
        print(f"Number of negative values in '{col}': {negative_count}")

# Clip negative values to zero
data[['Confirmed', 'Deaths', 'Recovered', 'Active']] = data[['Confirmed', 'Deaths', 'Recovered', 'Active']].clip(lower=0)

# Fix rows where "Deaths + Recovered > Confirmed"
data.loc[data['Deaths'] + data['Recovered'] > data['Confirmed'], 'Confirmed'] = data['Deaths'] + data['Recovered']

# Recalculate the "Active" column
data['Active'] = data['Confirmed'] - (data['Deaths'] + data['Recovered'])

# Final check for inconsistencies
inconsistencies = data[data['Active'] != calculated_active]
print(f"Number of inconsistencies after final correction: {len(inconsistencies)}")

# Check for negative values again
for col in numeric_columns:
    negative_count = (data[col] < 0).sum()
    print(f"Final check - number of negative values in '{col}': {negative_count}")

# Selecting the 0 values in Active
rows_to_drop = data[(data['Active'] == 0)]

# Deleting the rows where Active == 0
data = data.drop(rows_to_drop.index)

print("Veri kümesinin yeni şekli:", data.shape)
