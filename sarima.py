import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Veri setini yükleyin
file_path = 'final_data.csv'
data = pd.read_csv(file_path)

# 'Last_Update' sütununu datetime formatına dönüştürün
data['Last_Update'] = pd.to_datetime(data['Last_Update'], errors='coerce', infer_datetime_format=True)

# Dönüştürülemeyen tarih değerlerini kaldırın
data = data.dropna(subset=['Last_Update'])

# Belirli bir ülke için filtreleyin (örneğin, Arnavutluk)
country_data = data[data['Country_Region'] == 'Albania']

# Tarihe göre onaylanmış vakaları bir araya getirin
time_series_data = country_data.groupby('Last_Update')['Confirmed'].sum().reset_index()

# Tarihe göre sırala
time_series_data = time_series_data.sort_values(by='Last_Update')

# Zaman serisi analizi için indeksi ayarlayın ve sıklığı belirleyin
time_series_data.set_index('Last_Update', inplace=True)
time_series_data = time_series_data.asfreq('D', method='ffill')  # Eksik günleri doldur

# SARIMA modelini eğitin
sarima_model = SARIMAX(time_series_data['Confirmed'],
                       order=(1, 1, 1),  # ARIMA düzeni (p, d, q)
                       seasonal_order=(1, 1, 1, 7),  # SARIMA düzeni (P, D, Q, s)
                       enforce_stationarity=False,
                       enforce_invertibility=False).fit()

# Sonraki 30 günü tahmin edin
forecast_steps = 5
forecast = sarima_model.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=time_series_data.index[-1], periods=forecast_steps + 1, freq='D')[1:]
forecast_mean = forecast.predicted_mean

# Geçmiş verileri ve tahmini grafiğe dökün
plt.figure(figsize=(12, 6))
plt.plot(time_series_data['Confirmed'], label='Geçmiş Veriler')
plt.plot(forecast_index, forecast_mean, label='Tahmin', linestyle='--')
plt.title('Arnavutluk Onaylanmış Vakalarının SARIMA Tahmini')
plt.xlabel('Tarih')
plt.ylabel('Onaylanmış Vakalar')
plt.legend()
plt.grid(True)
plt.show()
