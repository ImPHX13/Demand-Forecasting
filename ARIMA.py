#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.dates as md
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import host_subplot
#import mpl_toolkits.axisartist as AA

#Import Dataset

df1 = pd.read_csv('B66S95.csv',parse_dates=True, dayfirst=True) 

df1.head()

df1['Date'] = pd.to_datetime(df1['Date'], format='%d/%m/%y')   

import cufflinks as cf         
import plotly
import plotly.offline as py
import plotly.graph_objs as go

cf.go_offline() 
py.init_notebook_mode()

#Plot of Quantity vs Date

df1.iplot(                         
    x='Date',
    y='Quantity',
    xTitle='Date',
    yTitle='Quantity',
    mode='markers',
    title='B66_SKU_95 Quantity')


df1['Date'] = pd.to_datetime(df1['Date'], format='%d/%m/%y')

#Plot of Monthly quantity sold

df1.set_index('Date').resample('M')['Quantity'].sum().plot(kind='bar',figsize=(5,5))    
plt.xlabel('Month', fontsize=18)
plt.ylabel('Quantity', fontsize=18)
plt.title('B66_SKU_95 Quantity')

#Outlier Detection using IQR and Boxplots

from numpy import percentile

q25, q75 = percentile(df1['Quantity'], 25), percentile(df1['Quantity'], 75)
iqr = q75 - q25

print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))

cut_off = iqr * 1.5
lower, upper = q25 - cut_off, q75 + cut_off

outliers = [x for x in df1['Quantity'] if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))

non_outliers = [x for x in df1['Quantity'] if x >= lower and x <= upper]
print('Non-outlier observations: %d' % len(non_outliers))

df1['Quantity'].count()
df1['Quantity'].iplot(kind='box', title='Box Plot of Quantity')

df1=df1.set_index('Date')

#Seasonal Decomposition

from random import randrange
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
series = df1['Quantity']
result = seasonal_decompose(series, model='multiplicative',period=7)
result.plot()
pyplot.show()

#ADF and KPSS to check for stationarity

from statsmodels.tsa.stattools import adfuller, kpss
def adf_test(timeseries):
    
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


adf_test(df1['Quantity'])

def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)

kpss_test(df1['Quantity'])

#Dataset divided into training and testing set

import numpy as np 
train_set, test_set= np.split(df1, [int(.70 *len(df1))])
train_set = train_set.astype('double')

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

#Simple Moving Average

y_hat_avg = test_set.copy()
y_hat_avg['MA Forecast'] = train_set['Quantity'].rolling(7).mean().iloc[-1]
plt.figure(figsize=(16,8))
plt.plot(train_set['Quantity'], label='Train')
plt.plot(test_set['Quantity'], label='Test')
plt.plot(y_hat_avg['MA Forecast'], label='Moving Average Forecast')
plt.legend(loc='best')
plt.title('Moving Average')
plt.show()

y_hat_avg['MA Forecast']

#RMSE and MAPE calculation

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test_set.Quantity, y_hat_avg['MA Forecast']))
print('RMSE = '+str(rmse))
abs_error = np.abs(test_set['Quantity']-y_hat_avg['MA Forecast'])
actual = test_set['Quantity']
mape = np.round(np.mean(abs_error/actual),4)
print('MAPE = '+str(mape))
resultsDf = pd.DataFrame({'Method':['Moving Average'], 'RMSE': [rmse],'MAPE':[mape]})
resultsDf

#Simple Exponential Smoothing

y_hat_avg = test_set.copy()
fit2 = SimpleExpSmoothing(np.asarray(train_set['Quantity'])).fit(smoothing_level=1,optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(test_set))
plt.figure(figsize=(16,8))
plt.plot(train_set['Quantity'], label='Train')
plt.plot(test_set['Quantity'], label='Test')
plt.plot(y_hat_avg['SES'], label='SES')
plt.legend(loc='best')
plt.title('Single Exponential Smoothing')
plt.show()

y_hat_avg['SES']

rms = sqrt(mean_squared_error(test_set.Quantity, y_hat_avg.SES))
print('RMSE = '+str(rms))
abs_error = np.abs(test_set['Quantity']-y_hat_avg.SES)
actual = test_set['Quantity']
mape = np.round(np.mean(abs_error/actual),4)
print('MAPE = '+str(mape))

tempResultsDf = pd.DataFrame({'Method':['SES'], 'RMSE': [rms],'MAPE': [mape] })
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Method', 'RMSE', 'MAPE']]
resultsDf

#Double Exponential Smoothing

y_hat_avg = test_set.copy()


fit1 = ExponentialSmoothing(np.asarray(train_set['Quantity']) ,seasonal_periods=5 ,trend='add', seasonal=None).fit()
y_hat_avg['DES'] = fit1.forecast(len(test_set))

plt.figure(figsize=(16,8))
plt.plot(train_set['Quantity'], label='Train')
plt.plot(test_set['Quantity'], label='Test')
plt.plot(y_hat_avg['DES'], label='DES')
plt.legend(loc='best')
plt.title('Double Exponential Smoothing')
plt.show()

y_hat_avg['DES']

rms = sqrt(mean_squared_error(test_set.Quantity, y_hat_avg.DES))
print('RMSE = '+str(rms))
abs_error = np.abs(test_set['Quantity']-y_hat_avg.DES)
actual = test_set['Quantity']
mape = np.round(np.mean(abs_error/actual),4)
print('MAPE = '+str(mape))

#RMSE and MAPE results displayed

tempResultsDf = pd.DataFrame({'Method':['DES'], 'RMSE': [rms],'MAPE': [mape]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf = resultsDf[['Method', 'RMSE', 'MAPE']]
resultsDf.head(10)

