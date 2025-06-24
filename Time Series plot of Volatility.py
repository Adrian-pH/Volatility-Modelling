# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 10:59:50 2025

@author: Adrian.ph689
"""

import datetime as dt
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------
# Downloading stock data
# ----------------------

end = dt.datetime.now()
start = end - dt.timedelta(days=365)
df = yf.download('BRK-B', start=start, end=end)
Close = df['Close']

# -------------------------------------
# Moving average estimate of volatility
# -------------------------------------

def moving_window(S,N):
    """
    Parameters
    ----------
    S : pandas Series - series of stock prices
    N : int - window size for the movign average

    Returns
    -------
    pandas Series - series of rolling averaged volatility
    """
    # Calculating daily returns
    R = (S.shift(-1) - S) / S
    R = R.dropna()

    # Calculating rollinng average of volatility
    sigma = R.rolling(window=N).std()
    sigma = sigma.dropna()
    annualised_sigma = sigma * 252**0.5 #annualising
    
    return annualised_sigma

# ---------------------------------------------
# Exponentially weighted movign average estimate of volatility
# ---------------------------------------------

def EWMA(S, alpha):
    """
    Parameters
    ----------
    S     : pandas Series - series of stock prices
    alpha : float - weighting factor (0 < alpha < 1)

    Returns
    -------
    pandas Series - series of exponentially weighted moving average (ewma) volatility
    """
    # Calculating daily returns
    R = (S.shift(-1) - S) / S
    R = R.dropna()
    
    # Calculating ewma of volitility
    R_squared = R**2
    var = R_squared.ewm(alpha=alpha, adjust=False).mean()
    sigma = var**0.5
    annualised_sigma = sigma * 252**0.5 #annualising
    
    return annualised_sigma
    
# -------------------------------
# Plotting volatility-time graphs
# -------------------------------

# Rolling average graph

plt.figure()
rolling_vol_3 = moving_window(Close, 3)
rolling_vol_7 = moving_window(Close, 7)
plt.plot(rolling_vol_3.index, rolling_vol_3.values, label='3-day rolling avg.')
plt.plot(rolling_vol_7.index, rolling_vol_7.values, label='7-day rolling avg.')
plt.title("Rolling Average of Daily Volatility")
plt.xlabel("Date")
plt.ylabel("Rolling Avg. Volatility")
plt.legend()
plt.tight_layout()
plt.show()


# EWMA graph

alphas = [0.3, 0.5, 0.7]

plt.figure()
for alpha in alphas:
    ewma = EWMA(Close, alpha)
    plt.plot(ewma.index, ewma.values, label=f"Alpha = {alpha}")
plt.title("Exonentially Weighted Moving Average of Daily Volatility")
plt.xlabel("Date")
plt.ylabel("EWMA Volatility")
plt.legend()
plt.tight_layout()
plt.show()

# EWMA vs Rolling average graph

plt.figure()
plt.plot(rolling_vol_3.index, rolling_vol_3.values, label='3-day rolling avg.')
ewma = EWMA(Close, 0.5)
plt.plot(ewma.index, ewma.values, label="Alpha = 0.5")
plt.title("Rolling Average vs EWMA of Daily Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.tight_layout()
plt.show()
