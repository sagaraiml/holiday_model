# -*- coding: utf-8 -*-
"""
Created on Nov 2019

@author: Sagar_Paithankar
"""

#importing libraries
import os
path = r'G:\Anaconda_CC\spyder\_client_xxx_my'
os.chdir(path)
import copy
import pandas as pd
import numpy as np
from datetime import *
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from holiday_support import *
from scipy.optimize import leastsq
from sklearn.metrics import mean_absolute_error
import copy

#reading from csv
holiday = pd.read_csv('holidays.csv', parse_dates=list(range(0,22)), infer_datetime_format=True)

df = holiday.copy()
#np.ravel(df.columns)
"""['Republic Day', 'MiladunNabi', 'Holi', 'Rama Navami',
       'Mahavir Jayanti', 'Good Friday', 'Buddha Purnima',
       'Independence Day', 'Janmashtami', 'Idul Fitr',
       'Mahatma Gandhi Jayanti', 'Dussehra', 'Deewali', 'Idul Juha',
       'Guru Nanak Jayanti', 'Muharram/Ashura', 'Christmas',
       'Maharishi Valmiki', 'Chhath Puja', 'Bank Holiday',
       'Maha Shivratri', "Children's Day"]"""


#event name we wont forcast for
ename = "Republic Day"
#day
Event = list(df[ename].dt.date)[1:]
#dayname for each day of event
Event_ = [x.strftime("%A") for x in Event]
Event_DT = dict(zip(Event, Event_))

#get all the load data for each holiday
#get weather data of holiday
#apply SVR
wdf = pd.DataFrame()
urd = pd.DataFrame()
for i in range(0,7):
    tday, day_7, Ns = get_event_data(i, Event)
    lf=tday.join(Ns['load'], rsuffix='Ns').join(day_7['load'], rsuffix='7')
    lf['ramp'] = lf['load'] - lf['load'].shift(1)
    lf['dow'] = lf.datetime.dt.dayofweek
    lf['weekend'] = np.where(lf['dow'] > 5, 1, 0)
    lf['month'] = lf.datetime.dt.month
    lf['tb'] = lf.datetime.apply(lambda x : ((x.hour*60 + x.minute)//15+1))
    print('load', end='>>')
    urd = urd.append(lf)
    urd.reset_index(inplace=True,drop=True)
    wd = actual(str(Event[i]),str(Event[i] + timedelta(days=1)))
    wd = wd.loc[:95, ['datetime', 'apparent_temperature','dew_point', 'wind_speed']]
    print(len(wd))
    wdf = wdf.append(wd)
    wdf.reset_index(inplace=True,drop=True)
    print('weather')

#year wise changes
for col in ['apparent_temperature','dew_point', 'wind_speed']:
    print('adding weather changes')
    wdf['delta1y'+'_'+col[:4]] = wdf[col] - wdf[col].shift(96)
    wdf['deltal2y'+'_'+col[:4]] = wdf[col] - wdf[col].shift(192)


#>> Include load 
#weather plot
#wd = forecast(str(Event[-1]),str(Event[-1]))
#wdf = wdf.append(wd)
#wdf = wdf[['datetime', 'apparent_temperature','dew_point', 'wind_speed']]
#fig, axs = plt.subplots(4)
#axs[0].plot(urd['load'],color='r', label='load')
#axs[0].legend(loc='upper left')
#axs[1].plot(wdf['apparent_temperature'],color='b', label='apparent_temperature')
#axs[1].legend(loc='upper left')
#axs[2].plot(wdf['dew_point'],color='g', label='dew_point')
#axs[2].legend(loc='upper left')
#axs[3].plot(wdf['wind_speed'],color='brown', label='wind_speed')
#axs[3].legend(loc='upper left')

#merging weather and load params
ndf = pd.merge(urd, wdf, how='inner', on='datetime')
ndf.reset_index(inplace=True, drop=True)
ndf['loadNs'] = ndf['loadNs'].shift(-96)
ndf['load7'] = ndf['load7'].shift(-96)
ndf['Target'] = ndf['load'].shift(-96)
ndf1 = ndf.dropna()
desc = ndf.describe()
#select featues
feature = ['load', 'loadNs', 'load7', 'ramp', 'dow', 'weekend',\
           'month', 'tb', 'apparent_temperature', 'dew_point', 'wind_speed',\
           'delta1y_appa', 'deltal2y_appa', 'delta1y_dew_', 'deltal2y_dew_',\
           'delta1y_wind', 'deltal2y_wind']

from sklearn.preprocessing import StandardScaler
srx = StandardScaler()
x = srx.fit_transform(ndf1[feature])
y = copy.deepcopy(ndf1['Target'])

from sklearn.model_selection import train_test_split, GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

import statsmodels.api as sm
X = sm.add_constant(x_train) # adding a constant
model = sm.OLS(y_train, X).fit() 
print(model.summary())

Xl = list(range(len(X.T)))
Xl.remove(4)
#removed 'ramp'
X = X[:,Xl]
model = sm.OLS(y_train, X).fit() 
print(model.summary())

Xl = list(range(len(X.T)))
Xl.remove(12)
#removed 'deltal2y_appa'
X = X[:,Xl]
model = sm.OLS(y_train, X).fit() 
print(model.summary())

Xl = list(range(len(X.T)))
Xl.remove(6)
#removed 'month'
X = X[:,Xl]
model = sm.OLS(y_train, X).fit() 
print(model.summary())

feature = ['load', 'loadNs', 'load7', 'dow', 'weekend',\
           'tb', 'apparent_temperature', 'dew_point', 'wind_speed',\
           'delta1y_appa', 'delta1y_dew_', 'deltal2y_dew_',\
           'delta1y_wind', 'deltal2y_wind']

#===================================================================================================
'''
from sklearn.svm import SVR
parameters = {'epsilon':[0.01, 0.1, 0.5, 1, 10, 25, 50, 100, 1000],\
              'C':[650,500,670,700,800]\
#              'degree':[3, 7, 11],\
              }
model = SVR(kernel='linear', gamma='auto')
clf = GridSearchCV(model, parameters, cv=3)
clf.fit(x_train, y_train)
clf.best_estimator_

print('Hello model'  )
model = clf.best_estimator_
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

a = mean_absolute_error(y_test, y_pred)
a*100/ np.mean(y_test)
'''
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>inference

#make daterange of day which to forecast
tdy1 = pd.DataFrame(pd.date_range(pd.Timestamp(Event[-1]), pd.Timestamp(Event[-1]) + timedelta(days=1), freq='15min'), columns=['datetime'])
tdy1 = tdy1.iloc[:96,]
#get previous yrs load data
load, day_7, Ns = get_event_data(-2, Event)
#merge daterange with previous yrs load
lfi=tdy1.join(load['load'])
#merge daterange with this years
tday, day_7, Ns = get_event_data(-1, Event)
lfi = lfi.join(Ns['load'], rsuffix='Ns').join(day_7['load'], rsuffix='7')

lfi['ramp'] = lfi['load'] - lfi['load'].shift(1)
lfi['dow'] = lfi.datetime.dt.dayofweek
lfi['weekend'] = np.where(lfi['dow'] > 5, 1, 0)
lfi['month'] = lfi.datetime.dt.month
lfi['tb'] = lfi.datetime.apply(lambda x : ((x.hour*60 + x.minute)//15+1))

lfi.loc[[0],'ramp'] = float(lfi.loc[[1],'ramp']) + (float(lfi.loc[[1],'ramp']) - float(lfi.loc[[2],'ramp']))


#weather
wdfi = pd.DataFrame()

wdi = actual(str(Event[-3]),str(Event[-3] + timedelta(days=1)))
wdi = wdi.loc[:95, ['datetime', 'apparent_temperature','dew_point', 'wind_speed']]
print(len(wdi))
wdfi = wdfi.append(wdi)

wdi = actual(str(Event[-2]),str(Event[-2] + timedelta(days=1)))
wdi = wdi.loc[:95, ['datetime', 'apparent_temperature','dew_point', 'wind_speed']]
print(len(wdi))
wdfi = wdfi.append(wdi)

wdi = forecast(str(Event[-1]),str(Event[-1] + timedelta(days=1)))
wdi = wdi.loc[:95, ['datetime', 'apparent_temperature','dew_point', 'wind_speed']]
print(len(wdi))
wdfi = wdfi.append(wdi)

wdfi.reset_index(inplace=True,drop=True)

for col in ['apparent_temperature','dew_point', 'wind_speed']:
    wdfi['delta1y'+'_'+col[:4]] = wdfi[col] - wdfi[col].shift(96)
    wdfi['deltal2y'+'_'+col[:4]] = wdfi[col] - wdfi[col].shift(192)

wtdf = wdfi.dropna().reset_index(drop=True)

xp = pd.merge(lfi, wtdf, how='inner', on='datetime')
xp = xp.iloc[:,1:]

##========
xp =xp[feature]
xp = srx.fit_transform(xp)
xp = sm.add_constant(xp)
tday['pred'] = model.predict(xp)

100*(np.abs(tday['load'] - tday['pred']).mean())/np.mean(tday['load'])

tday['Sunday'] = lfi['loadNs']
tday['Weekback'] = lfi['load7']
tday.plot(x='datetime',title = ename)

##========================Find Event day accuracy===================================================
##load day ahead accuracy
#acc = pd.read_csv('dayahead_accuracy_live.csv', usecols=[0,4,5], parse_dates=[0], infer_datetime_format=True)
##load holidday csv
#holiday = pd.read_csv('holidays.csv', parse_dates=list(range(0,22)), infer_datetime_format=True)
#holiday.dtypes
##recent data from 2018 (bcoz accuracy is tracked from 2018)
#dts = [pd.Timestamp(x).date() for x in np.array(holiday.iloc[7:9,:]).ravel()]
##holiday accuracy
#hacc = {}
#for x in dts :
#    if list(acc[acc['date'] == x]['ape']):
#        hacc.update({pd.Timestamp(x):list(acc[acc['date'] == pd.Timestamp(x)]['ape'])[0]})
##event name
#day = {}
#for i in range(len(dts)):
#        try:
#            en = holiday.columns[(holiday == pd.Timestamp(dts[i])).iloc[7]] [0]
#        except:
#            en = holiday.columns[(holiday == pd.Timestamp(dts[i])).iloc[8]] [0]
#        day.update({pd.Timestamp(dts[i]):en})
#
#hacc = pd.DataFrame(hacc, index=['mape']).T.reset_index()
#day = pd.DataFrame(day, index=['day']).T.reset_index()
##make event_mape 
#event_mape = pd.merge(day, hacc, how='inner', on='index')
#event_mape.columns = ['datetime', 'Event', 'Mape']
#event_mape = event_mape.set_index('datetime').sort_index().reset_index()
#event_mape.to_excel('Event_accuracy.xlsx', index=False)
##===================================================================================================
#===================plot all Ns and weekAgo load ===================================================
#holiday = pd.read_csv('holidays.csv', parse_dates=list(range(0,22)), infer_datetime_format=True)
#os.chdir(r'G:\Anaconda_CC\spyder\_client_xxx_my\past')
#dts = [pd.Timestamp(x).date() for x in np.array(holiday.iloc[7:9,:]).ravel()]
#for i in range(1, len(dts) - 1):
#    tday, day_7, Ns = get_event_data(i, dts)
#    lf=tday.join(Ns['load'], rsuffix='Ns').join(day_7['load'], rsuffix='7')
#    try:
#        en = holiday.columns[(holiday == pd.Timestamp(dts[i])).iloc[7]] [0]
#    except:
#        en = holiday.columns[(holiday == pd.Timestamp(dts[i])).iloc[8]] [0]
#    lf.plot(x='datetime', y=['load', 'loadNs', 'load7'],title = en+'_'+str(dts[i]))
#    figure = plt.gcf() # get current figure
#    figure.set_size_inches(12, 7)
#    plt.savefig(f"{en}"+'_'+str(dts[i])+'.png', dpi = 100)
#    print('saving image')
#    plt.close()
#===================================================================================================
"""
#===================================================================================================
#coefficient dictionary
coefficient = {}

#
def residuals(p, y):
    err = y - (p[0] * day_7 + p[1]* Ns)
    return err

for i in range(0,7):
    #Getting Eventdays Load, WeekAgo Load and Nearest Sunday Load
    tday, day_7, Ns = get_event_data(i, Event)
    #Nearest Sunday Load
    Ns = Ns['load']
    #Weekago Load
    day_7 = day_7['load']
    x = np.arange(1,97)
    #Actual Load
    y_true = tday['load']
    #random initialization coefficient
    p = [10,15]
    #new coefficient
    iter_coeff = leastsq(residuals, p, args=(y_true))
    print(list(iter_coeff[0]))
    #update 
    coefficient.update({Event[i] : list(iter_coeff[0])})

a = pd.DataFrame(coefficient, index=['c1', 'c2']).T
a.loc['mean'] = a.mean()


b = a.reset_index(drop=True).iloc[:-1, :]
s = pd.Series([0.1, 0.2, 0.1, 0.1, 0.2, 0.15, 0.15 ])
b = b.mul(s, axis=0)
b.loc['wt'] = b.sum()
list(b.loc['wt'])
y_pred = list(b.loc['wt'])[0] * day_7['load'] + list(b.loc['wt'])[1] *Ns['load']


#mape = {}
#
#for i in range(0,7):
#    tday, day_7, Ns = get_event_data(i, Event)
#    y_pred = list(a.loc['mean'])[0] * day_7['load'] + list(a.loc['mean'])[1] *Ns['load']
#    y_act = pd.Series(tday['load'])
#    mape.update({Event[i] : (mean_absolute_error(y_act, y_pred)*100/y_act.mean())})
#
#mdf = pd.DataFrame(mape, index=['mape']).T
#mdf['mape'] = 1 - mdf['mape']/100
#mdf['wts'] = mdf['mape'] / sum(mdf['mape'])
#
#x = a.reset_index(drop=True).iloc[:-1, :]
#b = x.mul(pd.Series(list(mdf['wts'])), axis=0)
#b.loc['wt'] = b.sum()
#list(b.loc['wt'])
#
#tday, day_7, Ns = get_event_data(-1, Event)
#y_pred = list(b.loc['wt'])[0] * day_7['load'] + list(b.loc['wt'])[1] *Ns['load']
#y_act = pd.Series(tday['load'])
#mape.update({Event[-1] : (mean_absolute_error(y_act, y_pred)*100/y_act.mean())})
#
#y_pred = list(a.loc['mean'])[0] * day_7['load'] + list(a.loc['mean'])[1] *Ns['load']
#mape.update({'mean' : (mean_absolute_error(y_act, y_pred)*100/y_act.mean())})



#import matplotlib.pyplot as plt
#plt.plot(x,y_pred)
#plt.title('Least-squares fit to noisy data {a}'.format(a=Event[5]))
#plt.legend(['Fit'])
#plt.show()
#===================================================================================================
"""