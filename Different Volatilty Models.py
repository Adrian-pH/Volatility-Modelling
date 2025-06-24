# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 14:54:50 2025

@author: Adrian.ph689
"""

import datetime as dt
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------
# Downloading stock data
# ----------------------

end = dt.datetime.now()
start = end - dt.timedelta(days=365)
df = yf.download('BRK-B', start=start, end=end)
Open = df['Open']
Close = df['Close']
High = df['High']
Low = df['Low']

# ------------------------------------------
# Traditional Close-to-Close rolling measure
# ------------------------------------------

def CC(Close, window):
    
    """
    Parameters
    ----------
    Close  : pandas Series - series of stock closing prices
    window : int - window size for the moving average

    Returns
    -------
    pandas Series - series of rolling annualised averaged volatility
    """
    # Calculating log returns
    log_R = np.log(Close / Close.shift(1))
    
    # Calculating the close-to-close average of volatility
    vol  =  log_R.rolling(window=window).std()
    vol = vol.dropna()
    annualised_vol = vol * np.sqrt(252) #annualising
    
    return annualised_vol

# ---------------------------------
# Parkinson (1980) Volatility Model
# ---------------------------------

def Parkinson(High, Low, window):
    
    """
    Parameters
    ----------
    High   : pandas Series - series of stock daily highs
    Low    : pandas Series - series of stock daily lows
    window : int - window size for the moving average

    Returns
    -------
    pandas Series - series of rolling annualised averaged volatility
    """
    # Calculating log High/Low ratio
    log_HL = np.log(High / Low) ** 2

    # Calculating the average volatility
    var = log_HL.rolling(window=window).mean()
    var = var.dropna()
    vol = np.sqrt(var / (4 * np.log(2)))
    annualised_vol = vol * np.sqrt(252) #annualising
    
    return annualised_vol

# --------------------------------------
# Garman & Klass (1980) Volatility Model
# --------------------------------------

def Garman_Klass(High, Low, Open, Close, window):
    
    """
    Parameters
    ----------
    High   : pandas Series - series of stock daily highs
    Low    : pandas Series - series of stock daily lows
    Open   : pandas Series - series of stock daily opens
    Close  : pandas Series - series of stock daily closes
    window : int - window size for the moving average

    Returns
    -------
    pandas Series - series of rolling annualised averaged volatility
    """
    # Calculating log ratios
    term_1 = 0.511 * (np.log(High / Low) ** 2)
    term_2 = -0.019 * np.log(Close / Open) * np.log(High * Low / Open ** 2)
    term_3 = -2 * np.log(High / Open) * np.log(Low / Open)

    # Calculating the average volatility
    var = (term_1 + term_2 + term_3).rolling(window=window).mean()
    var = var.dropna()
    vol = np.sqrt(var)
    annualised_vol = vol * np.sqrt(252) #annualising
    
    return annualised_vol

# -----------------------------------------
# Rogers & Satchell (1991) Volatility Model
# -----------------------------------------

def Rogers_Satchell(High, Low, Open, Close, window):
    
    """
    Parameters
    ----------
    High   : pandas Series - series of stock daily highs
    Low    : pandas Series - series of stock daily lows
    Open   : pandas Series - series of stock daily opens
    Close  : pandas Series - series of stock daily closes
    window : int - window size for the moving average

    Returns
    -------
    pandas Series - series of rolling annualised averaged volatility
    """
    # Calculating log ratios
    term_1 = np.log(High / Close) * np.log(High / Open)
    term_2 = np.log(Low / Close) * np.log(Low / Open)

    # Calculating the average volatility
    var = (term_1 + term_2).rolling(window=window).mean()
    var = var.dropna()
    vol = np.sqrt(var)
    annualised_vol = vol * np.sqrt(252) #annualising
    
    return annualised_vol

# -------------------------------
# Plotting volatility-time graphs
# -------------------------------

window = 7 # window size for all plots

# Close to Close
CC_vol_7 = CC(Close, window)
plt.plot(CC_vol_7.index, CC_vol_7.values, label='Close-to-Close', linestyle='--', color='blue')

# Parkinson
P_vol_7 = Parkinson(High, Low, window)
plt.plot(P_vol_7.index, P_vol_7.values, label='Parkinson (1980)', linestyle='-', color='green')

# Garman & Klass
GK_vol_7 = Garman_Klass(High, Low, Open, Close, window)
plt.plot(GK_vol_7.index, GK_vol_7.values, label='Garman & Klass (1980)', linestyle='-.', color='red')

# Rogers & Satchell
RS_vol_7 = Rogers_Satchell(High, Low, Open, Close, window)
plt.plot(RS_vol_7.index, RS_vol_7.values, label='Garman & Klass (1980)', linestyle='-', color='black')

plt.title("Different Volatility Models")
plt.xlabel("Date")
plt.ylabel("Rolling Avg. Volatility (Annualised)")
plt.legend()
plt.tight_layout()
plt.show()