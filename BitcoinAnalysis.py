import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime

# Function to fetch Bitcoin historical data from CoinGecko API 
def fetch_bitcoin_data_from_coingecko(start_date, end_date):
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    params = {
        'vs_currency': 'usd',
        'from': start_date.timestamp(),
        'to': end_date.timestamp()
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        prices['date'] = pd.to_datetime(prices['timestamp'], unit='ms')
        prices.set_index('date', inplace=True)
        return prices.drop(columns='timestamp')
    else:
        response.raise_for_status()


start_date = datetime.strptime("01-01-2022", "%d-%m-%Y")
end_date = datetime.strptime("01-01-2023", "%d-%m-%Y")

# Fetches  Bitcoin data for the specified time period
bitcoin_data = fetch_bitcoin_data_from_coingecko(start_date, end_date)

# Calculates the Moving Average
bitcoin_data['SMA_30'] = bitcoin_data['price'].rolling(window=30).mean()  # 30-day simple moving average

# Visualization of Price with Moving Average
plt.figure(figsize=(12, 6))
plt.plot(bitcoin_data.index, bitcoin_data['price'], label='Bitcoin Price (USD)')
plt.plot(bitcoin_data.index, bitcoin_data['SMA_30'], label='30-day SMA', color='orange')
plt.title('Bitcoin Price with 30-Day SMA')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Linear Regression
X = np.arange(len(bitcoin_data)).reshape(-1, 1)
X = sm.add_constant(X)
y = bitcoin_data['price']

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

# Makes the predictions for future trends
future_days = 30
future_dates = pd.date_range(start=end_date, periods=future_days + 1, freq='D')[1:]  # Exclude the start_date
future_X = sm.add_constant(np.arange(len(bitcoin_data), len(bitcoin_data) + future_days))
future_predictions = results.predict(future_X)

# Print future predictions
print("Future Predictions:")
for date, prediction in zip(future_dates, future_predictions):
    print(f"{date.date()}: ${prediction:.2f}")

# Visualize future trends
plt.figure(figsize=(12, 6))
plt.plot(bitcoin_data.index, bitcoin_data['price'], label='Historical Data')
plt.plot(future_dates, future_predictions, label='Future Predictions', linestyle='--', color='red')
plt.title('Bitcoin Future Trends')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
