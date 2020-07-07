# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:23:18 2020

@author: JEFF
"""
import numpy as np
import pandas as pd

data = pd.read_csv("./trivago//train.csv")
del data['user_id'], data['platform'], data['city']
del data['device'], data['current_filters'], data['prices']
head = data.head(100)

data = data[data.action_type == "clickout item"]
del data['action_type'], data['step'] 
head = data.head(100)
#%%

