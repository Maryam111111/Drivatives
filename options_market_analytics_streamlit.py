"""
Options Market Analytics - Streamlit UI Version
File: options_market_analytics_streamlit.py

What it does:
- Simulates option chain data without external API calls
- Displays tables for calls and puts in a Streamlit UI
- Computes put-call parity deviation
- Estimates historical volatility from generated price series
- Calculates Black-Scholes Greeks and implied volatility using a pure-Python bisection
- Generates interactive plots with Streamlit and matplotlib

How to run:
1) Create a virtualenv and install dependencies:
   pip install streamlit pandas numpy matplotlib

2) Run:
   streamlit run options_market_analytics_streamlit.py

Notes:
- This version uses Streamlit for UI and does not require yfinance or internet access.
- The option data and underlying prices are randomly generated for demonstration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import math
import matplotlib.pyplot as plt

# ------------------------- Utility functions -------------------------

def _ndist(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _ndist_pdf(x):
    return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

def bs_price(option_type, S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'call':
        return S * _ndist(d1) - K * math.exp(-r * T) * _ndist(d2)
    else:
        return K * math.exp(-r * T) * _ndist(-d2) - S * _ndist(-d1)

def bs_greeks(option_type, S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        delta = 1.0 if (option_type == 'call' and S > K) else 0.0
        delta = -1.0 if (option_type == 'put' and S < K) else delta
        return {'delta': delta, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = _ndist_pdf(d1)
    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * pdf_d1 * math.sqrt(T) / 100.0
    if option_type == 'call':
        delta = _ndist(d1)
        theta = (-S * pdf_d1 * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * _ndist(d2)) / 365.0
    else:
        delta = -_ndist(-d1)
        theta = (-S * pdf_d1 * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * _ndist(-d2)) / 365.0
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}

def implied_vol_bisection(option_type, market_price, S, K, T, r, tol=1e-6, max_iter=100):
    if market_price <= 0 or T <= 0:
        return np.nan
    low, high = 1e-6, 5.0
    for i in range(max_iter):
        mid = 0.5 * (low + high)
        price_mid = bs_price(option_type, S, K, T, r, mid)
        if abs(price_mid - market_price) < tol:
            return mid
        if price_mid > market_price:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)

# ------------------------- Streamlit UI -------------------------

def main():
    st.title("Options Market Analytics Demo")

    ticker = st.text_input("Ticker symbol", value="DEMO")
    np.random.seed(42)
    S = 100.0  # Simulated spot price
    r = 0.05
    expirations = [dt.date.today() + dt.timedelta(days=30)]
    expiry_choice = expirations[0]
    T = 30 / 365.0

    strikes = np.arange(90, 111, 2)
    calls = pd.DataFrame({'strike': strikes, 'lastPrice': [max(S - K + np.random.randn(), 0) for K in strikes]})
    puts = pd.DataFrame({'strike': strikes, 'lastPrice': [max(K - S + np.random.randn(), 0) for K in strikes]})

    merged = calls.merge(puts, on='strike', suffixes=('_call','_put'))
    merged['parity_diff'] = merged['lastPrice_call'] - merged['lastPrice_put'] - (S - merged['strike'] * math.exp(-r*T))

    st.subheader(f"Simulated Spot Price: {S:.2f}")
    st.subheader("Put-Call Parity Deviations")
    st.dataframe(merged[['strike','parity_diff']])

    # Historical volatility
    dates = pd.date_range(end=dt.date.today(), periods=252)
    prices = S * np.exp(np.cumsum(np.random.normal(0, 0.01, size=len(dates))))
    hist = pd.DataFrame({'Close': prices}, index=dates)
    hist['log_ret'] = np.log(hist['Close'] / hist['Close'].shift(1))
    rolling_window = 21
    hist['rolling_vol'] = hist['log_ret'].rolling(rolling_window).std() * np.sqrt(252)

    st.subheader("Rolling Annualized Volatility")
    st.line_chart(hist['rolling_vol'])

if __name__ == '__main__':
    main()
