# -*- coding: utf-8 -*-
"""
Created on Nov 2019

@author: Sagar_Paithankar
"""
import os
path = r'G:\Anaconda_CC\spyder\_client_xxx_my'
os.chdir(path)
import copy
import pandas as pd
import numpy as np
from datetime import *
from datetime import datetime, timedelta
from client_xxx_support import *
import matplotlib.pyplot as plt
import pymysql

# def get_data():
#     db_connection = pymysql.connect(host="139.59.42.147",user="pass",password="pass123",db="energy_consumption")   
#     SQL = "SELECT * FROM delhi_holidays"
#     df = pd.read_sql(SQL, con=db_connection)
#     return df

# df = get_data()


# holidays = df.copy()

holiday = pd.read_csv('holidays.csv', parse_dates=list(range(0,22)), infer_datetime_format=True)

Event = 'Guru Nanak Jayanti'
Dussehra = list(holiday[Event].dt.date)[-6:-1]
Dussehra_ = [x.strftime("%A") for x in Dussehra]

event = df[df['holiday_name'] == 'Guru Nanak Jayanti']
event['dayname'] = df.date.apply(lambda x : x.strftime("%A"))


Event = list(event['date'])
Event_ = list(event['dayname'])

Event_DT = dict(zip(Event, Event_))

Event = Event[:-1]
Event_ = [x.strftime("%A") for x in Event]


def doit(i):
    one = Event[i]
    tday= load().get_urd_data(str(one), str(one)).drop_duplicates('datetime')
    print('date on event is >>  ',one)
    print('day on event is >>  ',one.strftime("%A"))
    day7 = copy.deepcopy(one)
    day_7 = load().get_urd_data(str(one-timedelta(days=7)), str(one-timedelta(days=7))).drop_duplicates('datetime')
    print('7 minus the day is >>  ', list((day_7.datetime.dt.weekday_name).unique()))
    
    doy = one.strftime("%A")
    
    if doy == 'Monday' :
        one = one-timedelta(days=1)
    elif doy == 'Tuesday' :
        one = one-timedelta(days=2)
    elif doy == 'Wednesday' :
        one = one-timedelta(days=3)
    elif doy == 'Thursday' :
        one = one-timedelta(days=4)
    elif doy == 'Friday' :
        one = one-timedelta(days=5)
    elif doy == 'Saturday' :
        one = one-timedelta(days=6)
    else:
        one = one-timedelta(days=2)
        Nf_ = load().get_urd_data(str(one), str(one)).drop_duplicates('datetime')#if sunday take friday
        print((one - timedelta(7)).strftime("%A"))
    
    print('nearest suday we got or not >> ',(one).strftime("%A"))
    Ns = load().get_urd_data(str(one), str(one)).drop_duplicates('datetime')
    
    return tday, day_7, Ns

#===================================================================================================
import numpy as np
from scipy.optimize import leastsq


def residuals(p, y, x):
    err = y - (p[0] * day_7 + p[1]* Ns)
    return err

def peval(x, p):
    return p[0] * day_7 + p[1]* Ns

#change 'range()' this according to event
d = {}
for i in range(2):
    tday, day_7, Ns = doit(i)
    
    Ns = Ns['load']
    day_7 = day_7['load'] 
    x = np.arange(1,97)
    y_true = tday['load']
    y_meas = y_true + np.random.randn(len(x))
    p = [10,15]
    
    plsq = leastsq(residuals, p, args=(y_true, x))
    print(list(plsq[0]))
    
    d.update({Event[i] : list(plsq[0])})

a = pd.DataFrame(d, index=['c1', 'c2']).T
#a.loc['median'] = a.median()
#a.loc['mean'] = a.mean() >> Do not use it
"""
	c1	c2
2013-11-03	0.8016763507121198	0.14003437212010056
2016-10-30	0.8507053039821493	0.02671485119750372
"""


#calculating coeffient
mape = {}
for i in range(2):
    tday, day_7, Ns = doit(i)
    y_pred = list(a.median()[0] * day_7['load'] + a.median()[1] *Ns['load'])
    comp = pd.DataFrame()
    comp['Actual'] = tday['load']
    comp['Forecast'] = y_pred
    comp['Ns'] = Ns['load']
    comp['mae'] = np.abs(comp['Actual'] - comp['Forecast'])
    comp.loc['mean'] = comp.mean()
    mape.update({Event[i] : 100*comp.loc['mean','mae']/comp.loc['mean','Actual']})
#    mape.update({Event[i] : 100*(1 - comp.loc['mean','mae']/comp.loc['mean','Actual'])})


mapedf = pd.DataFrame(mape, index=['Mape']).round(3).T

"""
              Mape   inverse>>wts
2013-11-03  7.2269   0.1383 >>0.35
2016-10-30  3.8955   0.2567 >>0.65
"""
#mapedf.loc[:, 'sum'] = mapedf.sum(axis=1)
#mapedf.loc['Percent', :] = 100*np.divide(mapedf.loc['Mape', :],mapedf.sum(axis=1))
#wts = pd.DataFrame(index=mapedf.index)
#wts['wts'] = [0.1904, 0.1693, 0.0596, 0.0824, 0.1117, 0.0714, 0.3150]

#multiplying 
b = a.reset_index(drop=True)
s = pd.Series([0.35, 0.65])
b = b.mul(s, axis=0)
b.loc['wt'] = b.sum()
list(b.loc['wt'])

#prediction (7.11 >> median) (7.78 >> exclude 2016 and 2013)
tday, day_7, Ns = doit(1)
y_pred1= list(b.loc['wt'])[0] * day_7['load'] + list(b.loc['wt'])[1] *Ns['load']
y_pred2 = list(b.loc['wt'])[0] *Ns['load'] + list(b.loc['wt'])[1] *day_7['load'] 

y_pred = (y_pred1 + y_pred2)/2

comp = pd.DataFrame()
comp['Actual'] = tday['load']
comp['Forecast'] = y_pred
comp['Ns'] = Ns['load']
comp['mae'] = np.abs(comp['Actual'] - comp['Forecast'])

x = range(1, 97)
plt.plot( x, 'Actual', data=comp, marker='', color='blue')
plt.plot( x, 'Forecast', data=comp, marker='', color='red')
plt.plot( x, 'Ns', data=comp, marker='', color='yellow')
plt.legend()

comp.loc['mean'] = comp.mean()
