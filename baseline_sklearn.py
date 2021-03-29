#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss

import pandas as pd
import numpy as np
np.random.seed(0)

from tqdm.notebook import tqdm

import os

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# print(os.path.dirname(os.path.abspath('Bioinf590_TabNet.ipynb')))

GOOGLE_DRIVE_PATH_AFTER_MYDRIVE = 'Bioinf 590/Final Project'
GOOGLE_DRIVE_PATH = os.path.join('drive', 'My Drive', GOOGLE_DRIVE_PATH_AFTER_MYDRIVE)
print('In Working Directory', os.listdir(GOOGLE_DRIVE_PATH))

DATA_PATH = os.path.join(GOOGLE_DRIVE_PATH, 'lish-moa')

train = pd.read_csv(DATA_PATH+"/train_features.csv")
train_target = pd.read_csv(DATA_PATH+'/train_targets_scored.csv')
test = pd.read_csv(DATA_PATH+'/test_features.csv')


# In[ ]:


all_columns = set(train.columns)
print(len(all_columns))
remove_columns = set(['sig_id', 'cp_type', 'cp_time', 'cp_dose'])
train_columns = all_columns - remove_columns
print(len(train_columns))

x_train = train[train_columns]
print(x_train.shape)

# target remove sig_id
y_train = train_target[set(train_target.columns) - set(['sig_id'])]
print('shape of y:',y_train.shape)


# In[ ]:


# decision tree regression
from sklearn.tree import DecisionTreeRegressor
import time
x = x_train[0:10000]
y = y_train[0:10000]
x_val = x_train[10000:15000]
y_val = y_train[10000:15000]
print(x.shape,y.shape,x_val.shape,y_val.shape)
start_time = time.time()
model = DecisionTreeRegressor(criterion='mae') #max_features='sqrt',max_depth=10,
model.fit(x,y)
print('model trained in time', (time.time()-start_time)/60)


# In[ ]:


start_time = time.time()
y_pred = model.predict(x_val)
print('y_pred are done in :',time.time()-start_time)


# In[ ]:


# from sklearn.metrics import log_loss
# log_loss(y_val,y_pred)


# In[ ]:




