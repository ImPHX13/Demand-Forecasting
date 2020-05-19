#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#Import dataset

df = pd.read_csv('data.csv',parse_dates=True, dayfirst=True)   

df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')

print(df.dtypes)
df.head()
df=df.set_index('Date')
df.index

#Create a timeseseries

ts=df['Quantity']   
ts.head()

#Rolling mean and standard deviation calculation to check for stationarity

rolling_mean = ts.rolling(window = 5).mean()
rolling_std = ts.rolling(window = 5).std()
plt.plot(ts, color = 'blue', label = 'Original')
plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
plt.legend(loc = 'best')
plt.title('Rolling Mean & Rolling Standard Deviation')
plt.show()

#ADF test for checking stationarity of timeseries

result = adfuller(ts)
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))

#Timeseries log transformation

ts_log = np.log(ts) 
plt.plot(ts_log)

result = adfuller(ts_log)
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))

#Function for ADF test

def get_stationarity(timeseries):
    
    
    rolling_mean = timeseries.rolling(window=5).mean()
    rolling_std = timeseries.rolling(window=5).std()
    
    
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    result = adfuller(timeseries)
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))  

rolling_mean = ts_log.rolling(window=5).mean()
ts_log_minus_mean = ts_log - rolling_mean
ts_log_minus_mean.dropna(inplace=True)
get_stationarity(ts_log_minus_mean)


#Exponential Decay

rolling_mean_exp_decay = ts_log.ewm(halflife=5, min_periods=0, adjust=True).mean()
ts_log_exp_decay = ts_log - rolling_mean_exp_decay
ts_log_exp_decay.dropna(inplace=True)
get_stationarity(ts_log_exp_decay)

#Timeseries log shifted to make it stationary

ts_log_shift = ts_log - ts_log.shift()
ts_log_shift.dropna(inplace=True)
get_stationarity(ts_log_shift)

#Timeseries log differenced to make it stationary

ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)

ts_log_diff.dropna(inplace=True)
get_stationarity(ts_log_diff)

#Seasonal Decomposition to check for seasonality and trends

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log,freq=7)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
get_stationarity(ts_log_decompose)

#ACF and PACF plots to find p and q values

from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

#Gridsearch for ideal p,q parameters based on lowest AIC value

import statsmodels.api as sm
resDiff = sm.tsa.arma_order_select_ic(ts_log, max_ar=7, max_ma=7, ic='aic', trend='c')
print('ARMA(p,q) =',resDiff['aic_min_order'],'is the best.')

#Fitting ARIMA model from the obtained (p,d,q) values

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(ts_log, order=(1, 1, 0))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())

#Bring back the predictions to original scale

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())


predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()

#Plot of Actual vs Forecasted values

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('ARIMA   MAPE: %.4f'% np.mean(np.abs(predictions_ARIMA-ts)/np.abs(ts)))

#RMSE and MAPE calculations
mape = np.mean(np.abs(predictions_ARIMA - ts)/np.abs(ts))  
rmse = np.mean((predictions_ARIMA - ts)**2)**.5  
print(mape)
print(rmse)

#Summary of ARIMA model
results_ARIMA.summary()




