import pandas as pd
import numpy as np
import datetime as dt
from sklearn import metrics
import matplotlib.pyplot as plt
import statsmodels.api as sm
#sys.path.append('C:\\Users\YOONHO\\Desktop\\pyodbc\\')
#sys.path.append('C:\\Users\\YOONHO\\Desktop\\analysis\\salesbyLoc\\')
import os
#path = 'C:\\Users\\ykimmate14\\Downloads\\career_doc\\verisk\\analysis\\salesbyloc'
path = 'C:\\Users\\YOONHO\\Desktop\\analysis\\qtysoldanalysis'
os.chdir(path)

#==============================================================================
### SQL query that's equivalanet of the csv files
# obqty_qry = '''
#     select INVDATE, CUSTOMER, Category, ITEM, qty, SHPCITY, SHPSTATE
#     	from KOALA.dbo.v_AllSalesbyComponent
# '''
# dimdate_qry = '''
# select [Date], [Year], [Month], WeekOfYear, [Quarter]
# 	,MonthOfQuarter, WeekOfQuarter, WeekOfMonth, IsHolidayUSA, IsHolidayWeek, MondayOfWeek
# 	from KOALA.dbo.DimDate
# '''
#==============================================================================


obqty_csv = 'v_allsalesbycomponent.csv'

obqty_df = pd.read_csv(obqty_csv, encoding = 'ISO-8859-1'
                 ,dtype = {'INVDATE': str
                           ,'CUSTOMER': str
                           ,'Category': str
                           ,'ITEM': str
                           ,'qty': int
                           ,'SHPCITY':str
                           ,'SHPSTATE':str})


dimdate_csv = 'DimDate.csv'
date_df = pd.read_csv(dimdate_csv)

state_csv = 'StateRegion.csv'
state_df = pd.read_csv(state_csv)

#merging ob with date data
df_raw = obqty_df.merge(date_df, how = 'inner', left_on = ['INVDATE'], right_on = ['Date'])
#merging state data
df_raw = df_raw.merge(state_df, how = 'inner', left_on = ['SHPSTATE'], right_on = ['State'])

top_item = ['BPM-SB-14Q-F'
            ,'OLC-FMS-1000Q'
            ,'OLC-FMS-1000F'
            ,'BPM-MFT-4Q'
            ,'BPM-FMS-8T'
            ,'EB-FMS-0600Q'
            ,'BPM-SB-14F-F'
            ,'EB-FMS-0600T'
            ,'EB-FMS-0600F'
            ,'BPM-FMS-8Q']
df = df_raw[['INVDATE','FirstDayOfMonth','MondayOfWeek','CUSTOMER','Month','Year', 'WeekOfYear','Category','ITEM', 'qty', 'EastOrWest']]

#converting date from string to datetime
df['INVDATE'] = df['INVDATE'].apply(lambda x: dt.datetime.strptime(x, '%m/%d/%Y'))
df['FirstDayOfMonth'] = df['FirstDayOfMonth'].apply(lambda x: dt.datetime.strptime(x, '%m/%d/%Y'))
df['MondayOfWeek'] = df['MondayOfWeek'].apply(lambda x: dt.datetime.strptime(x, '%m/%d/%Y'))

#wrtie the df to csv
#df.to_csv('C:\\Users\\YOONHO\\Desktop\\analysis\\salesbyLoc\\soldqty_region.csv', index = False)

#excluding AMZ DIRECT
df = df[df.CUSTOMER != 'AMZ DIRECT']

#total sales aggregated by year and month
x = df.groupby(['Year','Month'], as_index = False)['qty'].sum()
x = x.sort_values(['Year','Month'])
#excluding this year - 2018
x = x[x.Year != 2018]

p1 = plt.figure()
plt.ylabel('ShippedQty')
plt.xlabel('Month')
for yr in x.Year.unique():
    plt.plot(x[x.Year == yr].Month, x[x.Year == yr].qty, label = yr)
plt.legend()
plt.show()

#by region
df_reg = df.groupby(['FirstDayOfMonth', 'EastOrWest'], as_index = False)['qty'].sum()
df_reg = df_reg.sort_values(by=['FirstDayOfMonth'])
p1 = plt.figure()
plt.ylabel('Qty sold')
plt.xlabel('Month')
plt.xticks(rotation = 90)
for region in df_reg.EastOrWest.unique():
    plt.plot(df_reg[df_reg.EastOrWest == region].FirstDayOfMonth, df_reg[df_reg.EastOrWest == region].qty, label = region)
plt.legend()
plt.title('Qty sold East vs West')
plt.show()

#group by month and Category
df_cat = df.groupby(['FirstDayOfMonth', 'Category'], as_index = False)['qty'].sum()
p2 = plt.figure()
plt.ylabel('Qty sold')
plt.xlabel('Month')
plt.xticks(rotation = 90)
for cat in df_cat.Category.unique():
    plt.plot(df_cat[df_cat.Category == cat].FirstDayOfMonth, df_cat[df_cat.Category == cat].qty, label = cat)
plt.legend()
plt.title('Qty sold by category')
plt.show()

#by category and region
df_catreg = df.groupby(['FirstDayOfMonth', 'Category', 'EastOrWest'], as_index = False)['qty'].sum()
for cat in df_catreg.Category.unique():
    p3 = plt.figure()
    temp_df = df_catreg[df_catreg.Category == cat]
    for region in temp_df.EastOrWest.unique():
        plt.plot(temp_df[temp_df.EastOrWest == region].FirstDayOfMonth, temp_df[temp_df.EastOrWest == region].qty, label = region)
    plt.legend()
    plt.title('Qty sold by region - %s'%cat)
    plt.xticks(rotation = 90)
    plt.show()
    


#by category and region - excluding AMZDR
df_catreg = df[df.CUSTOMER != 'AMZ DIRECT']
df_catreg = df_catreg.groupby(['FirstDayOfMonth', 'Category', 'EastOrWest'], as_index = False)['qty'].sum()
for cat in df_catreg.Category.unique():
    p3 = plt.figure()
    temp_df = df_catreg[df_catreg.Category == cat]
    for region in temp_df.EastOrWest.unique():
        plt.plot(temp_df[temp_df.EastOrWest == region].FirstDayOfMonth, temp_df[temp_df.EastOrWest == region].qty, label = region)
    plt.legend()
    plt.title('Qty sold by region - %s'%cat)
    plt.xticks(rotation = 90)
    plt.show()


from statsmodels.tsa.stattools import adfuller



#testing stationary
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries, titlenote):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation - %s'%titlenote)
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
'''
#by item and week
cat = 'MATTRESS'
#df_1 = df[df.ITEM ==item].groupby(['MondayOfWeek'], as_index = False)['qty'].sum()
df_1 = df[df.Category ==cat].groupby(['MondayOfWeek'], as_index = False)['qty'].sum()

df_1 = df_1.set_index('MondayOfWeek')
df_1['log_qty'] = np.log(df_1.qty)
tsdf = df_1.qty
test_stationarity(tsdf, item)


#transform data into stationary
#take log transformation to penalize high values compared to low values
tsdflog = df_1.log_qty
moving_avg = pd.rolling_mean(tsdflog,12)
plt.plot(tsdflog)
plt.plot(moving_avg, color='red')

ts_log_moving_avg_diff = tsdflog - moving_avg
ts_log_moving_avg_diff.head(12)

ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff, item)

sm.tsa.seasonal_decompose(df_1.qty, model = 'additive')

#use differencing to transform data into stationary
ts_log_diff = tsdflog - tsdflog.shift()
plt.plot(ts_log_diff)
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff, item)
'''

#by category and month - TS decomposition
cat = 'MATTRESS'
df_cat = df.groupby(['MondayOfWeek', 'Category'], as_index = False)['qty'].sum()
dfcatmon = df_cat[df_cat.Category == cat][['MondayOfWeek','qty']]
dfcatmon = dfcatmon.set_index('MondayOfWeek')
res = sm.tsa.seasonal_decompose(dfcatmon.qty, model = 'additive')
res.plot()
trend = res.trend
seasonal = res.seasonal
residual =res.resid.dropna()

#plot ts components
plt.plot(seasonal)
plt.xticks(rotation = 90)
plt.title('seasonal component')

ts_cbrt_decompose = np.cbrt(residual)
ts_cbrt_decompose.dropna(inplace=True)
test_stationarity(ts_cbrt_decompose, cat)

#ARIMA
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_cbrt_decompose, nlags=20)
lag_pacf = pacf(ts_cbrt_decompose, nlags=20, method='ols')

plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_cbrt_decompose)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_cbrt_decompose)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_cbrt_decompose)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_cbrt_decompose)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

from statsmodels.tsa.arima_model import ARIMA
#p = AR parameter, partial autocorrelation
#i
#q = MA parameter, autocorrelation
series_y = dfcatmon.qty.astype('float64')
model = ARIMA(ts_cbrt_decompose, order=(0,1,3))  
results_AR = model.fit(disp=-1)  
yhat_1 = results_AR.fittedvalues**3+trend+seasonal
plt.plot(series_y, label = 'y')
plt.plot(yhat_1, color='red', label = 'yhat')
plt.title('Actual Qty sold vs fitted value - MATTRESS Category')
plt.legend()

#fitting trend
import statsmodels.api as sm
trend = trend.dropna()
z = np.polyfit(np.arange(0,len(trend)), trend,1)
f = np.poly1d(z)
trend_yhat = f(np.arange(0,len(trend)))
plt.plot(trend.values, label = 'trend')
plt.plot(trend_yhat, label = 'linear fit on trend')
plt.title('Linear fit on trend: y = %s + %sx'%(str(round(z[1],2)),str(round(z[0],2))))
plt.legend()
plt.show()

df_yhat = pd.DataFrame()
df_yhat['MondayOfWeek'] = date_df['MondayOfWeek'].apply(lambda x: dt.datetime.strptime(x, '%m/%d/%Y')).drop_duplicates()
trend_df = trend.to_frame()
trend_df = trend_df.rename(columns= {'qty':'trend'})
df_yhat = df_yhat.merge(trend_df, how = 'left', left_on = 'MondayOfWeek',right_index = True)
df_yhat['week'] = df_yhat.MondayOfWeek.apply(lambda x: x.weekofyear)

seasonal_df = seasonal.to_frame()
seasonal_df['week'] = seasonal_df.apply(lambda x: x.index.weekofyear)
seasonal_df = seasonal_df[:53]
seasonal_df = seasonal_df.rename(columns= {'qty':'seasonal'})

arima_df = results_AR.fittedvalues.to_frame()
arima_df = arima_df.rename(columns= {0:'arimavalue'})
df_yhat = df_yhat.merge(arima_df, how = 'left', left_on = 'MondayOfWeek',right_index = True)


df_yhat = df_yhat.merge(seasonal_df, how = 'left', left_on = 'week', right_on = 'week')
df_yhat = df_yhat[(df_yhat.MondayOfWeek >= '2017-09-18') & (df_yhat.MondayOfWeek < '2018-06-01')]
df_yhat = df_yhat.reset_index()

#filling trend
import math
for i, row in df_yhat.iterrows():
    if math.isnan(row.trend):
        trend_v = df_yhat.trend[i-1]
        df_yhat.trend[i] = trend_v + f[1]
        
#filling arima
forecast = results_AR.forecast(steps = 36)[0]
for i, row in df_yhat.iterrows():
    if math.isnan(row.arimavalue):
        v = forecast[i-1]
        df_yhat.arimavalue[i] = v

df_yhat = df_yhat.set_index('MondayOfWeek')
yhat_2 = df_yhat.arimavalue**3+df_yhat.trend+df_yhat.seasonal
yhat = pd.concat([yhat_1, yhat_2])
plt.plot(series_y, label = 'y')
plt.plot(yhat, color='red', label = 'yhat')
plt.title('Actual Qty sold vs fitted value - MATTRESS Category')
plt.legend()
plt.xticks(rotation = 90)

yhat[yhat.index >= '2018-04-01'].sum()

plt.figure(figsize=(30, 5))
plt.plot(pd.DataFrame(forecast ** 3))
plt.title('Display the predictions with the ARIMA model')
plt.show()


yhat = results_AR.fittedvalues**3+trend+seasonal
plt.plot(series_y)
plt.plot(yhat, color='red')

plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-series_y[1:])**2))

model = ARIMA(series_y, order=(2,1,0))  
results_MA = model.fit(disp=-1)  
plt.plot(series_y)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-series_y)**2))

model = ARIMA(ts_log_decompose, order=(2, 1, 5))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_decompose)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_decompose[1:])**2))