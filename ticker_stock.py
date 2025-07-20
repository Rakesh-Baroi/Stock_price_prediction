import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# App config
st.set_page_config(page_title="📈 Stock Price Predictor", layout="centered")

# ---- Styled container with frame and shadow ----
st.markdown("""
    <style>
        .main-box {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            margin-top: 30px;
            margin-bottom: 30px;
        }
        .box-title {
            font-size: 22px;
            font-weight: bold;
            color: #333333;
            margin-bottom: 15px;
        }
    </style>
    <div class="main-box">
        <div class="box-title">📉 Stock Price Prediction</div>
""", unsafe_allow_html=True)

st.markdown("Predict the **closing price** of a stock for a future date (within the next 2 weeks).")

# ---- Ticker input ----
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT, TSLA)", value="AAPL").upper()

# ---- Load and preprocess data ----
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, period="5y", interval="1d", auto_adjust=False)
    df = df.dropna()
    df['Change'] = df['Close'] - df['Open']
    df['Range'] = df['High'] - df['Low']
    df['Prev_Close'] = df['Close'].shift(1)
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df

try:
    df = load_data(ticker)

    # ---- Show data and allow download ----
    st.markdown("## 📅 View & Download Historical Data")

    min_history_date = df.index.min().date()
    start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=90), min_value=min_history_date, max_value=datetime.today().date())
    end_date = st.date_input("End date", value=datetime.today(), min_value=start_date, max_value=datetime.today().date())

    filtered_df = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]

    if not filtered_df.empty:
        st.dataframe(filtered_df)

        csv = filtered_df.to_csv(index=True)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{ticker}_historical_data_{start_date}_to_{end_date}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No data available for the selected date range.")

    # ---- Prediction Section ----
    if filtered_df.shape[0] < 30:
        st.warning("Please select a wider date range (at least 30 business days) for accurate prediction.")
    else:
        df_for_prediction = filtered_df.copy()

        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change', 'Range', 'Prev_Close']
        X = df_for_prediction[features]
        y = df_for_prediction['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBRegressor()
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        st.markdown("## 📉 Past Predictions vs Actual")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_test.index, y_test.values, label='Actual', color='blue')
        ax.plot(y_test.index, y_pred, label='Predicted', color='orange')
        ax.set_title("Actual vs Predicted Close Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.legend()
        st.pyplot(fig)

        # ---- Future Prediction ----
        n_days = 10
        last_row = df_for_prediction.iloc[[-1]].copy()
        today_price = float(last_row['Close'].iloc[0])
        future_data = []
        future_dates = pd.date_range(start=datetime.today(), periods=n_days * 2, freq='B')[:n_days]

        for _ in range(n_days):
            new_row = last_row.copy()
            new_row['Prev_Close'] = last_row['Close']
            new_row['Change'] = last_row['Close'] - last_row['Open']
            new_row['Range'] = last_row['High'] - last_row['Low']
            features_input = new_row[features]
            scaled_input = scaler.transform(features_input)
            predicted_close = model.predict(scaled_input)[0]
            future_data.append(predicted_close)

            new_row['Open'] = predicted_close
            new_row['High'] = predicted_close * 1.01
            new_row['Low'] = predicted_close * 0.99
            new_row['Close'] = predicted_close
            new_row['Volume'] = last_row['Volume'].values[0]
            last_row = new_row

        future_price_map = dict(zip([d.date() for d in future_dates], future_data))

        min_date = datetime.today().date()
        max_date = min_date + timedelta(days=14)
        user_date = st.date_input("Choose a date within the next 2 weeks:", min_value=min_date, max_value=max_date)

        if st.button("🔮 Predict Price"):
            if user_date in future_price_map:
                predicted_price = future_price_map[user_date]
                change_from_today = predicted_price - today_price

                st.subheader("📊 Prediction Result")
                st.write(f"**Today's Close (from selected range):** ${today_price:.2f}")
                st.write(f"**Predicted Close on {user_date}:** ${predicted_price:.2f}")
                st.write(f"**Change from Today:** {'🔺' if change_from_today >= 0 else '🔻'} ${change_from_today:.2f}")
            else:
                st.error("Please choose a business day (weekday) within the next 2 weeks.")

except Exception as e:
    st.error(f"Failed to retrieve or process data for '{ticker}'. Make sure the ticker is correct.")

# ---- Close styled box ----
st.markdown("</div>", unsafe_allow_html=True)

# ---- Footer ----
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey; font-size: 0.9em;'>"
    "© 2025 Rakesh's prediction. All rights reserved."
    "</div>",
    unsafe_allow_html=True
)
