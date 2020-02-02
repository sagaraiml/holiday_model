# -*- coding: utf-8 -*-
"""
Created on Nov 2019

@author: Sagar_Paithankar
"""

import os
path = r'G:\Anaconda_CC\spyder\_client_xxx_my'
os.chdir(path)
import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
from math import ceil
import copy
from datetime import *
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def get_urd_data(sdt, tdt):
    table_name = 'client_xxx_data'
    db_connection = pymysql.connect(host="139.59.42.147",user="pass",password="pass123",db="energy_consumption")
    SQL = "SELECT * FROM " + table_name + " WHERE date BETWEEN '" + sdt + "' AND '" + tdt + "'"
    df = pd.read_sql(SQL, con=db_connection)
    df['datetime'] = pd.to_datetime(df.date.astype(str) + ' ' + df.time_slot.str[:5])
    df['date'] = df.datetime.dt.date
    df['load']=df.block_load
    df['tb'] = df.datetime.apply(lambda x : ((x.hour*60 + x.minute)//15+1))
    df = df[['datetime','date','load','tb']]
    return df

def doit(i, Event):
    one = Event[i]
    tday= get_urd_data(str(one), str(one)).drop_duplicates('datetime')
    print('day on event is >>  ',one.strftime("%A"))
    day7 = copy.deepcopy(one)
    day_7 = get_urd_data(str(one-timedelta(days=7)), str(one-timedelta(days=7))).drop_duplicates('datetime')
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
        Nf_ = get_urd_data(str(one), str(one)).drop_duplicates('datetime')#if sunday take friday
        print((one - timedelta(7)).strftime("%A"))
    
    print('nearest suday we got or not >> ',(one).strftime("%A"))
    Ns = get_urd_data(str(one), str(one)).drop_duplicates('datetime')
    
    return tday, day_7, Ns