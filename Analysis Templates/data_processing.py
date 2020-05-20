# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 11:40:29 2020

@author: Nicholas Sotiriou - github: @nsotiriou88 // nsotiriou88@gmail.com
"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb


# import dataset
df = pd.read_csv('data_set.csv', header=0)
cols = df.columns.to_list()[0].split(sep=';')
df = df[df.columns.to_list()[0]].str.split(';', expand=True)

for i, col in enumerate(cols):
    cols[i] = col.replace('"', '')

df.columns = cols


data_type = {'age': np.int8, 'duration': np.int16, 'campaign': np.int8,
             'pdays': np.int16, 'previous': np.int8, 'emp.var.rate':
            np.float32, 'cons.price.idx': np.float32, 'cons.conf.idx':
            np.float32, 'euribor3m': np.float32, 'nr.employed': np.float32,
            'y': np.int8}

categ_feat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
              'poutcome']

numer_feat = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx',
              'cons.conf.idx', 'euribor3m', 'nr.employed']

df['y'].replace(to_replace={'"yes"': 1, '"no"': 0}, inplace=True)
df = df.astype(dtype=data_type, errors='raise')




job_cat = pd.get_dummies(df['job'], prefix='job', drop_first=False)
job_cat.columns = [col.replace('"', '') for col in job_cat.columns]
job_cat.drop(['job_unknown'], axis=1, inplace=True)




marital_cat = pd.get_dummies(df['marital'], drop_first=True)
education_cat = pd.get_dummies(df['education'], drop_first=True)



job_cat = pd.get_dummies(df['job'], drop_first=True)
job_cat = pd.get_dummies(df['job'], drop_first=True)
job_cat = pd.get_dummies(df['job'], drop_first=True)
job_cat = pd.get_dummies(df['job'], drop_first=True)




df['nr.employed'].value_counts()
df['nr.employed'].describe()






