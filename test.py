# stockd_app.py

import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit

st.set_page_config(page_title="Stockd", layout="wide")
st.title("Stockd")

st.markdown("""
Upload your **Sales CSV** and **Stock CSV** to get stock recommendations for the next 7 days.
""")

# --- File upload ---
sales_file = st.file_uploader("Upload Sales file", type=["xlsx"])
stock_file = st.file_uploader("Upload Stock file", type=["xlsx"])

if sales_file and stock_file:
  # --- Load data ---
  sales_df = pd.read_excel(sales_file)
  stock_df = pd.read_excel(stock_file)

  # Ensure date is datetime
  sales_df['date'] = pd.to_datetime(sales_df['date'])
  stock_df['date'] = pd.to_datetime(stock_df['date'])

  # Encode product_name
  le = LabelEncoder()
  sales_df['product_encoded'] = le.fit_transform(sales_df['product_name'])
  stock_df['product_encoded'] = le.transform(stock_df['product_name'])

  # --- Aggregate daily sales per product ---
  daily_sales = sales_df.groupby(['date', 'product_encoded', 'product_name'])['sales'].sum().reset_index()

  # --- Create lag features for forecasting ---
  lag_features = [1, 7]  # lag 1 day and lag 7 days
  daily_sales = daily_sales.sort_values(['product_encoded', 'date'])

  for lag in lag_features:
      daily_sales[f'lag_{lag}'] = daily_sales.groupby('product_encoded')['sales'].shift(lag)
  
  # Drop initial rows with NaN lags
  daily_sales = daily_sales.dropna().reset_index(drop=True)

  # --- Features & target ---
  feature_cols = ['product_encoded'] + [f'lag_{lag}' for lag in lag_features]
  X = daily_sales[feature_cols]
  y = daily_sales['sales']

  # --- Train model (XGBoost) ---
  model = XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42)
  model.fit(X, y)

  # --- Forecast next 7 days per product ---
  forecast_days = 7
  results = []

  for product_code, product_name in zip(le.transform(le.classes_), le.classes_):
      product_data = daily_sales[daily_sales['product_encoded'] == product_code].copy()
      last_row = product_data.iloc[-1]
      
      lags = last_row[[f'lag_{lag}' for lag in lag_features]].values.tolist()
      forecast_qty = 0
      avg_daily_sales = product_data['sales'].mean()
      temp_lags = lags.copy()

      for day in range(forecast_days):
          x_input = [product_code] + temp_lags
          pred = model.predict(np.array(x_input).reshape(1, -1))[0]
          forecast_qty += pred
          # Update lags
          temp_lags = [pred] + temp_lags[:-1]

      # --- Get current stock ---
      latest_stock = stock_df[stock_df['product_encoded'] == product_code].sort_values('date').iloc[-1]
      remaining_stock = latest_stock['remaining_stock']

      # --- Apply smart stock rule ---
      days_of_stock = remaining_stock / avg_daily_sales if avg_daily_sales > 0 else float('inf')
      if days_of_stock < forecast_days:
          needed = forecast_qty - remaining_stock
          decision = f"BUY {needed} units"
          reason = (f"Forecasted demand for the next {forecast_days} days is {forecast_qty:.1f} units, "
                    f"but current stock is only {remaining_stock}. "
                    f"Stock will run out in ~{days_of_stock:.1f} days. Purchase {needed:.1f} units to avoid stockout.")
      else:
          needed = 0
          decision = "DON'T BUY"
          reason = (f"Forecasted demand for the next {forecast_days} days is {forecast_qty:.1f} units, "
                    f"and current stock is {remaining_stock}. "
                    f"Stock is sufficient for ~{days_of_stock:.1f} days. No purchase required.")

      results.append({
          'Product': product_name,
          'Decision': decision,
          'Qty_to_Buy': needed,
          'Reason': reason
      })

  final_df = pd.DataFrame(results)

  st.subheader("Stock Recommendations")
  st.dataframe(final_df)

  # Download button
  csv = final_df.to_csv(index=False)
  st.download_button(
      data=csv,
      file_name='stockd_report.csv',
      mime='text/csv'
  )

  from sklearn.model_selection import TimeSeriesSplit

  tscv = TimeSeriesSplit(n_splits=5)
  for train_index, test_index in tscv.split(X):
      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]

  model = XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  from sklearn.metrics import mean_absolute_error

  mae = mean_absolute_error(y_test, y_pred)
  print(f"MAE: {mae:.2f}")
  from sklearn.metrics import mean_squared_error

  rmse = mean_squared_error(y_test, y_pred, squared=False)
  print(f"RMSE: {rmse:.2f}")
  mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
  print(f"MAPE: {mape:.2f}%")

  import matplotlib.pyplot as plt

  plt.figure(figsize=(10,5))
  plt.plot(y_test.values, label='Actual', marker='o')
  plt.plot(y_pred, label='Predicted', marker='x')
  plt.title('Sales Forecast vs Actual')
  plt.xlabel('Time')
  plt.ylabel('Quantity Sold')
  plt.legend()
  plt.show()