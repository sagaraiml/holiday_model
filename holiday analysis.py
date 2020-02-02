# -*- coding: utf-8 -*-
"""
Created on Oct 2019

@author: Sagar_Paithankar
"""

import os
path = r'G:\Anaconda_CC\spyder\_client_xxx_my'
os.chdir(path)

import pandas as pd
import numpy as np
from datetime import *
from datetime import datetime, timedelta
from client_xxx_support import *
import matplotlib.pyplot as plt
import pymysql

#def get_data():
#    db_connection = pymysql.connect(host="139.59.42.147",user="pass",password="pass123",db="energy_consumption")   
#    SQL = "SELECT * FROM delhi_holidays"
#    df = pd.read_sql(SQL, con=db_connection)
#    return df
#df = get_data()
#holidays = df.copy()
#a  = df[df['holiday_name'] == 'Naraka Chaturdasi']
#Event = list(a['date'])
#Event_ = [x.strftime("%A") for x in Event]

df = pd.read_csv('holidays.csv', parse_dates=list(range(0,22)), infer_datetime_format=True)
holidays = df.copy()
print(list(holidays.columns))

ename = "Christmas"
filename = f'{ename}.xlsx'

Event  = df[ename].to_list()
Event_ = [x.strftime("%A") for x in Event]

Event_DT = dict(zip(Event, Event_))

Event = Event[-4:-1]
Event_ = [x.strftime("%A") for x in Event]

"""
every year ka 2 days before and 1 day after >> 4 days
for dussera 4 days befor 
"""

h2016 = load().get_urd_data(str(Event[0]-timedelta(days=3)), str(Event[0]+timedelta(days=2))).drop_duplicates('datetime')
h2017 = load().get_urd_data(str(Event[1]-timedelta(days=3)), str(Event[1]+timedelta(days=2))).drop_duplicates('datetime')
h2018 = load().get_urd_data(str(Event[2]-timedelta(days=3)), str(Event[2]+timedelta(days=2))).drop_duplicates('datetime')

dn18 = [x.strftime("%A") for x in h2018.datetime.dt.date.unique()]
dn17 = [x.strftime("%A") for x in h2017.datetime.dt.date.unique()]
dn16 = [x.strftime("%A") for x in h2016.datetime.dt.date.unique()]

h2016.date = h2016.date.astype('str')
h2017.date = h2017.date.astype('str')
h2018.date = h2018.date.astype('str')

p16 = pd.pivot(h2016, index='tb', columns='date', values='load')
p17 = pd.pivot(h2017, index='tb', columns='date', values='load')
p18 = pd.pivot(h2018, index='tb', columns='date', values='load')


with pd.ExcelWriter(filename) as writer:
    for df in [p18, p17, p16]:
        df = df.append(pd.Series(), ignore_index=True)
#        df.loc['mean'] = df.mean()
        df.loc['median'] = df.median()
        df.loc['max'] = df.max()
#        df.loc['std'] = df.std(axis=0)
        if (list(df.columns)[2])[:4] == '2018':
            df.loc['days'] = dn18
        if (list(df.columns)[2])[:4] == '2017':
            df.loc['days'] = dn17
        if (list(df.columns)[2])[:4] == '2016':
            df.loc['days'] = dn16
        df.reset_index(inplace=True)
        print('done for year : ', (list(df.columns)[2])[:4])
        df.to_excel(writer,sheet_name=list(df.columns)[2][:4], index=False)
writer.close()

cols = list(p16.columns)
x = range(1, 97)
plt.plot( x, cols[0], data=p16, marker='', color='blue')
plt.plot( x, cols[1], data=p16, marker='', color='green')
plt.plot( x, cols[2], data=p16, marker='', color='brown')
plt.plot( x, cols[3], data=p16, marker='', color='red')
plt.plot( x, cols[4], data=p16, marker='', color='black')
plt.plot( x, cols[5], data=p16, marker='', color='yellow')
plt.xlabel(list((h2016.datetime.dt.weekday_name).unique()))
plt.legend()
figure = plt.gcf() # get current figure
figure.set_size_inches(12, 7)
plt.savefig(f"{ename}_2016.png", dpi = 100)
plt.close()

cols = list(p17.columns)
x = range(1, 97)
plt.plot( x, cols[0], data=p17, marker='', color='blue')
plt.plot( x, cols[1], data=p17, marker='', color='green')
plt.plot( x, cols[2], data=p17, marker='', color='brown')
plt.plot( x, cols[3], data=p17, marker='', color='red')
plt.plot( x, cols[4], data=p17, marker='', color='black')
plt.plot( x, cols[5], data=p17, marker='', color='yellow')
plt.xlabel(list((h2017.datetime.dt.weekday_name).unique()))
plt.legend()
figure = plt.gcf() # get current figure
figure.set_size_inches(12, 7)
plt.savefig(f"{ename}_2017.png", dpi = 100)
plt.close()

cols = list(p18.columns)
x = range(1, 97)
plt.plot( x, cols[0], data=p18, marker='', color='blue')
plt.plot( x, cols[1], data=p18, marker='', color='green')
plt.plot( x, cols[2], data=p18, marker='', color='brown')
plt.plot( x, cols[3], data=p18, marker='', color='red')
plt.plot( x, cols[4], data=p18, marker='', color='black')
plt.plot( x, cols[5], data=p18, marker='', color='yellow')
plt.xlabel(list((h2018.datetime.dt.weekday_name).unique()))
plt.legend()
figure = plt.gcf() # get current figure
figure.set_size_inches(12, 7)
plt.savefig(f"{ename}_2018.png", dpi = 100)
plt.close()


##weather
#
#
#w2016 = weather().combiner1(str(Event[0]-timedelta(days=3))[:10], str(Event[0]+timedelta(days=2))[:10], 13, 8, 'darksky_actual').iloc[:, :2]
#w2016['tb'] = w2016.datetime.apply(lambda x : ((x.hour*60 + x.minute)//15+1))
#w2016['date'] = w2016.datetime.dt.date.astype('str')
#w2016 = pd.pivot(w2016, index='tb', columns='date', values='apparent_temperature').interpolate(method='time')
#
#w2017 = weather().combiner1(str(Event[1]-timedelta(days=3))[:10], str(Event[1]+timedelta(days=2))[:10], 13, 8, 'darksky_actual').iloc[:, :2]
#w2017['tb'] = w2017.datetime.apply(lambda x : ((x.hour*60 + x.minute)//15+1))
#w2017['date'] = w2017.datetime.dt.date.astype('str')
#w2017 = pd.pivot(w2017, index='tb', columns='date', values='apparent_temperature').interpolate(method='time')
#
#w2018 = weather().combiner1(str(Event[2]-timedelta(days=3))[:10], str(Event[2]+timedelta(days=2))[:10], 13, 8, 'darksky_actual').iloc[:, :2]
#w2018['tb'] = w2018.datetime.apply(lambda x : ((x.hour*60 + x.minute)//15+1))
#w2018['date'] = w2018.datetime.dt.date.astype('str')
#w2018 = pd.pivot(w2018, index='tb', columns='date', values='apparent_temperature').interpolate(method='time')
#
#cols = list(w2017.columns)
#x = range(1, 97)
#plt.plot( x, cols[0], data=w2017, marker='', color='blue')
#plt.plot( x, cols[1], data=w2017, marker='', color='green')
#plt.plot( x, cols[2], data=w2017, marker='', color='brown')
#plt.plot( x, cols[3], data=w2017, marker='', color='red')
#plt.plot( x, cols[4], data=w2017, marker='', color='black')
#plt.plot( x, cols[5], data=w2017, marker='', color='yellow')
#plt.xlabel(list((w2017.datetime.dt.weekday_name).unique()))
#plt.legend()
#figure = plt.gcf() # get current figure
#figure.set_size_inches(12, 7)
#plt.savefig(f"{ename}_2017.png", dpi = 100)
#plt.close()
#
#
#
#
#
