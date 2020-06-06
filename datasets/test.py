# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:26:57 2020

@author: JEFF
"""

import numpy as np
import pandas as pd
import csv

data = pd.read_pickle("./sample/train.txt")
#%%
data = pd.read_csv("sample_train-item-views.csv", delimiter=';')
#%%
data_y = pd.read_csv('./yoochoose-data/yoochoose-clicks.dat')
#%%
data_y.value_counts()
#dataset = './yoochoose-data/yoochoose-clicks.dat'
#with open(dataset, "r") as f:
#    reader = csv.DictReader(f, delimiter=',')
#%%
print(data.count())
a = data.item_id.value_counts()
b = data.session_id.value_counts()
#%%