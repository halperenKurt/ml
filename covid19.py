import pandas as pd
"""

# Load all the uploaded files
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

# Read and concatenate all files for train dataset
dataframes = [pd.read_csv(file_path) for file_path in file_paths_for_train]
final_data_df = pd.concat(dataframes, ignore_index=True)

# Read and concatenate all files for test dataset
test_dataframes = [pd.read_csv(file_path) for file_path in file_paths_for_test]
test_df = pd.concat(test_dataframes, ignore_index=True)

# Save the concatenated train dataframe to a CSV file
final_data_df.to_csv('final_data.csv', index=False)

# Save the concatenated test dataframe to a CSV file
test_df.to_csv('test_data.csv', index=False) 


"""

df = pd.read_csv("final_data.csv")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df_cleaned =df.dropna(subset=["Lat", "Long_", "Incident_Rate", "Case_Fatality_Ratio"])
print(df_cleaned.isnull().sum())
print(df_cleaned["Case_Fatality_Ratio"].tail(40))
