import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from statsmodels.tsa.arima.model import ARIMA
import warnings
from itertools import product


class DataCleaner:
    def __init__(self, train_joblib_path='final_data.joblib', test_joblib_path='test_data.joblib'):
        self.train_joblib_path = train_joblib_path
        self.test_joblib_path = test_joblib_path
        self.data = None
        self.test_data = None

    def load_or_create_data(self):
        if os.path.exists(self.train_joblib_path) and os.path.exists(self.test_joblib_path):
            self.data = joblib.load(self.train_joblib_path)
            self.test_data = joblib.load(self.test_joblib_path)
            print("Joblib files loaded.")
        else:
            # Paths for raw CSV files
            file_paths_for_train = [f'datasets/01-{str(i).zfill(2)}-2021.csv' for i in range(1, 31)]
            file_paths_for_test = [f'datasets/02-{str(i).zfill(2)}-2021.csv' for i in range(1, 6)]

            # Read and concatenate all train files
            dataframes = [pd.read_csv(file_path) for file_path in file_paths_for_train]
            self.data = pd.concat(dataframes, ignore_index=True)

            # Read and concatenate all test files
            test_dataframes = [pd.read_csv(file_path) for file_path in file_paths_for_test]
            self.test_data = pd.concat(test_dataframes, ignore_index=True)

            # Save the concatenated data to joblib files
            joblib.dump(self.data, self.train_joblib_path)
            joblib.dump(self.test_data, self.test_joblib_path)
            print("Joblib files created and saved.")

    def clean_data(self):
        if self.data is None or self.test_data is None:
            raise ValueError("Data is not loaded. Call 'load_or_create_data' before cleaning data.")
        self.data = self.data.dropna(subset=["Lat", "Long_", "Incident_Rate", "Case_Fatality_Ratio"])
        self.test_data = self.test_data.dropna(subset=["Lat", "Long_", "Incident_Rate", "Case_Fatality_Ratio"])
        self.data = self.data.drop(columns=["FIPS", "Admin2", "Province_State"], errors='ignore')
        self.test_data = self.test_data.drop(columns=["FIPS", "Admin2", "Province_State"], errors='ignore')

        # Fix "Active" column
        self.data['Active'] = self.data['Confirmed'] - (self.data['Deaths'] + self.data['Recovered'])
        self.test_data['Active'] = self.test_data['Confirmed'] - (
                self.test_data['Deaths'] + self.test_data['Recovered'])

        # Clip negative values
        numeric_columns = ['Confirmed', 'Deaths', 'Recovered', 'Active']
        self.data[numeric_columns] = self.data[numeric_columns].clip(lower=0)
        self.test_data[numeric_columns] = self.test_data[numeric_columns].clip(lower=0)

        # Ensure "Confirmed >= Deaths + Recovered"
        self.data.loc[self.data['Deaths'] + self.data['Recovered'] > self.data['Confirmed'], 'Confirmed'] = \
            self.data['Deaths'] + self.data['Recovered']
        self.test_data.loc[
            self.test_data['Deaths'] + self.test_data['Recovered'] > self.test_data['Confirmed'], 'Confirmed'] = \
            self.test_data['Deaths'] + self.test_data['Recovered']

        # Recalculate "Active"
        self.data['Active'] = self.data['Confirmed'] - (self.data['Deaths'] + self.data['Recovered'])
        self.test_data['Active'] = self.test_data['Confirmed'] - (
                self.test_data['Deaths'] + self.test_data['Recovered'])

    def create_target_variable(self):
        if self.data is None or self.test_data is None:
            raise ValueError("Data is not loaded. Call 'load_or_create_data' before creating target variable.")
        self.data['Target'] = (self.data['Active'] > 0).astype(int)
        self.test_data['Target'] = (self.test_data['Active'] > 0).astype(int)

    def get_cleaned_data(self):
        return self.data, self.test_data


class LogisticRegressionModel:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.model = LogisticRegression(max_iter=1000)
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def prepare_data(self):
        # Drop non-numeric columns
        self.train_data = self.train_data.select_dtypes(exclude=['object'])
        self.test_data = self.test_data.select_dtypes(exclude=['object'])

        # Prepare training and testing sets
        self.X_train = self.train_data.drop(columns=['Target'], errors='ignore')
        self.y_train = self.train_data['Target']
        self.X_test = self.test_data.drop(columns=['Target'], errors='ignore')
        self.y_test = self.test_data['Target']

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed.")

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))


class ForecastModel:
    def __init__(self, train_joblib_path='final_data.joblib', test_joblib_path='test_data.joblib'):
        self.train_joblib_path = train_joblib_path
        self.test_joblib_path = test_joblib_path
        self.train_data = None
        self.test_data = None

    def load_or_create_data(self):
        if os.path.exists(self.train_joblib_path) and os.path.exists(self.test_joblib_path):
            self.train_data = joblib.load(self.train_joblib_path)
            self.test_data = joblib.load(self.test_joblib_path)
            print("Joblib files loaded.")
        else:
            # Paths for raw CSV files
            train_files = [f'datasets/01-{str(i).zfill(2)}-2021.csv' for i in range(1, 31)]
            test_files = [f'datasets/02-{str(i).zfill(2)}-2021.csv' for i in range(1, 6)]

            # Read and concatenate train and test files
            train_dfs = [pd.read_csv(file) for file in train_files]
            test_dfs = [pd.read_csv(file) for file in test_files]

            self.train_data = pd.concat(train_dfs, ignore_index=True)
            self.test_data = pd.concat(test_dfs, ignore_index=True)

            # Save the concatenated data to joblib files
            joblib.dump(self.train_data, self.train_joblib_path)
            joblib.dump(self.test_data, self.test_joblib_path)
            print("Joblib files created and saved.")

    def clean_data(self):
        if self.train_data is None or self.test_data is None:
            raise ValueError("Data is not loaded. Call 'load_or_create_data' before cleaning data.")

        # Drop irrelevant columns and handle missing values
        cols_to_drop = ["FIPS", "Admin2", "Province_State"]
        self.train_data.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        self.test_data.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        self.train_data.dropna(subset=["Lat", "Long_", "Incident_Rate", "Case_Fatality_Ratio"], inplace=True)
        self.test_data.dropna(subset=["Lat", "Long_", "Incident_Rate", "Case_Fatality_Ratio"], inplace=True)

        # Recalculate "Active" column
        for df in [self.train_data, self.test_data]:
            df['Active'] = df['Confirmed'] - (df['Deaths'] + df['Recovered'])
            df['Active'] = df['Active'].clip(lower=0)
            df['Confirmed'] = np.maximum(df['Confirmed'], df['Deaths'] + df['Recovered'])

    def optimize_arima(self, time_series, p_values, d_values, q_values):
        """
        Optimize ARIMA parameters by grid search using AIC as the criterion.
        """
        warnings.filterwarnings("ignore", category=UserWarning)
        best_aic = float('inf')
        best_order = None
        best_model = None

        for p, d, q in product(p_values, d_values, q_values):
            try:
                model = ARIMA(time_series, order=(p, d, q))
                fitted_model = model.fit()
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_order = (p, d, q)
                    best_model = fitted_model
            except Exception:
                continue

        print(f"Best ARIMA order: {best_order} with AIC: {best_aic}")
        return best_model, best_order

    def forecast_for_country(self, country, forecast_steps=5):
        # Filter country-specific data
        train_country_data = self.train_data[self.train_data['Country_Region'] == country]
        test_country_data = self.test_data[self.test_data['Country_Region'] == country]

        # Process "Last_Update" column
        train_country_data.loc[:, 'Last_Update'] = pd.to_datetime(train_country_data['Last_Update']).dt.date
        test_country_data.loc[:, 'Last_Update'] = pd.to_datetime(test_country_data['Last_Update']).dt.date

        # Create time series
        train_time_series = train_country_data.groupby('Last_Update')['Confirmed'].sum()
        train_time_series = train_time_series.asfreq('D').interpolate(method='linear')

        test_time_series = test_country_data.groupby('Last_Update')['Confirmed'].sum()

        # Optimize ARIMA parameters
        p_values = range(0, 3)
        d_values = range(0, 2)
        q_values = range(0, 3)
        fitted_model, best_order = self.optimize_arima(train_time_series, p_values, d_values, q_values)

        # Forecast
        forecast_dates = pd.date_range(start=train_time_series.index[-1], periods=forecast_steps + 1, freq='D')[1:]
        forecast_values = fitted_model.forecast(steps=forecast_steps)
        forecast_series = pd.Series(forecast_values.values, index=forecast_dates)

        # Ensure consistency in index format
        test_time_series.index = pd.to_datetime(test_time_series.index).date
        forecast_series.index = pd.to_datetime(forecast_series.index).date

        # Visualize results
        plt.figure(figsize=(14, 7))
        plt.plot(train_time_series, label='Train Data', marker='o', color='blue')
        plt.plot(forecast_series, label='Forecast', linestyle='--', color='orange', marker='x')
        plt.scatter(test_time_series.index, test_time_series.values, color='red', label='Test Data', zorder=5)
        plt.title(f'ARIMA Forecast vs Test Data for Confirmed Cases in {country} (Order: {best_order})', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Confirmed Cases', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        # Numerical comparison
        comparison = pd.DataFrame({
            "Forecast": forecast_series,
            "Actual": test_time_series
        })
        print(comparison)

    def run_forecast(self, countries, forecast_steps=5):
        for country in countries:
            print(f"Processing forecast for {country}...")
            self.forecast_for_country(country, forecast_steps=forecast_steps)


class EDAAnalysis:
    def __init__(self, data, numeric_columns, date_column='Last_Update'):
        self.data = data
        self.numeric_columns = numeric_columns
        self.date_column = date_column

    def plot_histograms(self, log_scale=True, bins=30):
        """Plots colorful histograms for numeric columns."""
        colors = sns.color_palette("hsv", len(self.numeric_columns))

        for i, column in enumerate(self.numeric_columns):
            plt.figure(figsize=(10, 6))
            plt.hist(
                self.data[column],
                bins=bins,
                color=colors[i],
                alpha=0.7,
                edgecolor='black',
                log=log_scale
            )
            plt.title(f'Distribution of {column} (Log Scale)', fontsize=14)
            plt.xlabel(column, fontsize=12)
            plt.ylabel('Frequency (Log Scale)', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

    def plot_boxplots(self):
        """Applies log transformation to numeric columns and plots boxplots."""
        data_log_transformed = self.data.copy()
        for column in self.numeric_columns:
            data_log_transformed[column] = self.data[column].apply(lambda x: np.log1p(x) if x > 0 else 0)

        for column in self.numeric_columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=data_log_transformed[column], color='lightgreen')
            plt.title(f'Boxplot of {column} (Log Transformed)', fontsize=14)
            plt.xlabel(column, fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

    def time_series_analysis(self, end_date='2021-01-31'):
        """Performs time series analysis on the dataset."""
        try:
            # Convert the date column to datetime
            self.data[self.date_column] = pd.to_datetime(
                self.data[self.date_column],
                format='%Y-%m-%d %H:%M:%S',
                errors='coerce'
            )

            # Drop rows with invalid dates
            self.data = self.data.dropna(subset=[self.date_column])

            # Filter data by the specified end date
            filtered_data = self.data[self.data[self.date_column] <= end_date]

            # Group by date and aggregate numeric columns
            time_series_data = filtered_data.groupby(self.date_column)[['Confirmed', 'Deaths', 'Recovered']].sum()

            # Plot time series for each numeric column
            for column in ['Confirmed', 'Deaths', 'Recovered']:
                plt.figure(figsize=(12, 6))
                plt.plot(time_series_data.index, time_series_data[column], marker='o', linestyle='-', linewidth=1.5)
                plt.axvline(pd.to_datetime(end_date), color='red', linestyle='--', label='Data Ends')
                plt.title(f'Time Series of {column}', fontsize=14)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel(column, fontsize=12)
                plt.grid(axis='both', linestyle='--', alpha=0.7)
                plt.legend()
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"Error during time series analysis: {e}")


if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.load_or_create_data()
    cleaner.clean_data()
    cleaner.create_target_variable()
    train_data, test_data = cleaner.get_cleaned_data()
    test2_data = pd.read_csv("datasets/test2_data.csv")

    lr_model = LogisticRegressionModel(train_data, test_data)
    lr_model.prepare_data()
    lr_model.train_model()
    lr_model.evaluate_model()
    print(test_data.head())

    model = ForecastModel(train_joblib_path='final_data.joblib', test_joblib_path='test_data.joblib')
    model.load_or_create_data()
    model.clean_data()
    countries = ["US", "Russia", "Japan", "China", "Colombia"]
    model.run_forecast(countries, forecast_steps=5)

    numeric_columns = ['Confirmed', 'Deaths', 'Recovered', 'Active', 'Incident_Rate', 'Case_Fatality_Ratio']
    eda = EDAAnalysis(train_data, numeric_columns)
    eda.plot_histograms()
    eda.plot_boxplots()
    eda.time_series_analysis()
