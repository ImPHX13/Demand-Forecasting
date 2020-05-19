#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
from statsmodels.tsa.seasonal import seasonal_decompose      
from pmdarima import auto_arima                              

import warnings
warnings.filterwarnings("ignore")

#Loading the dataset

df = pd.read_csv('data.csv',parse_dates=True, dayfirst=True)

df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
df=df.set_index('Date')
df.head()
df.tail()
df1 = df

#Seasonal decomposition to check for seasonality and trends

result = seasonal_decompose(df1['Quantity'],freq=7)
result.plot();

#ADF test to check for stationarity

from statsmodels.tsa.stattools import adfuller

def adf_test(series,title=''):
    
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC')
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string()) 
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")

adf_test(df1['Quantity'])

#Auto ARIMA to find the values of (p,d,q)x(P,D,Q)

auto_arima(df1['Quantity'],seasonal=True,m=).summary()

#Splitting dataset into training and tresting dataset.

len(df1)

train = df1.iloc[:123]
test = df1.iloc[123:]

#Training the model using SARIMAX

model = SARIMAX(train['Quantity'],exog = train['ReturnQuantity'],order=(0,1,1),seasonal_order=(0,0,1,7),enforce_invertibility=False,enforce_stationarity=True)
results = model.fit()
results.summary()

start=len(train)
end=len(train)+len(test)-1
exog_forecast = test['ReturnQuantity']
predictions = results.predict(start=start, end=end, exog=exog_forecast).rename('SARIMAX(0,1,1)(0,0,1,7) Predictions')

#Plot of Actual vs Forecasted values

title='B74S117'
ylabel='Quantity'
xlabel='Date'

ax = test['Quantity'].plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
for x in test.query('holiday==1').index: 
    ax.axvline(x=x, color='k', alpha = 0.3);

#MSE and RMSE calculation

from statsmodels.tools.eval_measures import mse,rmse

error1x = mse(test['Quantity'], predictions)
error2x = rmse(test['Quantity'], predictions)


print(f'SARIMAX(2,1,2)(1,0,0,7) MSE Error: {error1x:11.10}')
print(f'SARIMAX(2,1,2)(1,0,0,7) RMSE Error: {error2x:11.10}')

#MAPE calculation

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(test['Quantity'],predictions)




