#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[29]:


train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
target = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
# train_drug = pd.read_csv('/kaggle/input/lish-moa/train_drug.csv')
train_drug = pd.read_csv('/kaggle/input/lish-moa/train_drug.csv')
print('read all the data into pandas')


# In[3]:


all_columns = set(train.columns)
print(len(all_columns))
remove_columns = set(['sig_id', 'cp_type', 'cp_time', 'cp_dose'])
train_columns = all_columns - remove_columns
print(len(train_columns))
target_columns = set(target.columns) - set(['sig_id'])
print(len(target_columns))


# In[25]:


def z_Score_norm(train):
  GENES = [col for col in train.columns if col.startswith('g-')]
  CELLS = [col for col in train.columns if col.startswith('c-')]
  control_data = train[train['cp_type'] == 0]
  control_24 = control_data[control_data['cp_time'] == 0] ##Control data for cp_time = 24
  control_48 = control_data[control_data['cp_time'] == 1]
  control_72 = control_data[control_data['cp_time'] == 2]

  for col in (GENES+CELLS):
    train.loc[train['cp_time'] == 0, col] -= control_24[col].mean()
    train.loc[train['cp_time'] == 0, col] /= control_24[col].std()

    train.loc[train['cp_time'] == 1, col] -= control_48[col].mean()
    train.loc[train['cp_time'] == 1, col] /= control_48[col].std()

    train.loc[train['cp_time'] == 2, col] -= control_72[col].mean()
    train.loc[train['cp_time'] == 2, col] /= control_72[col].std()

  return train


# In[5]:


class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size,dtype):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.5)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
        
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.5)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))
    
    def forward(self, x):
        
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.relu(self.dense1(x))
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.relu(self.dense2(x))
        
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        
        return x


# In[27]:


def train_model(train_data,val_data,model,learning_rate,weight_decay):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                              max_lr=1e-2, epochs=num_epochs, steps_per_epoch=len(train_data))
    # enumerate epochs
    for epoch in range(num_epochs):
        # enumerate mini batches
        epoch_loss = 0
        start_time = time.time()
        for i, (x, y) in enumerate(train_data):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y,y_pred)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss
        epoch_loss /= i
        epoch_time = (time.time() - start_time)/60
        print(f"EPOCH: {epoch}, train_loss: {epoch_loss}, in time{epoch_time}")
    
    # validation set
    # check validation accuracy
    # model.eval()
    final_val_loss = 0
    for i,(x,y) in enumerate(val_data):
        y_pred = model(x)
        loss = loss_fn(y,y_pred)
        final_val_loss += loss
    final_val_loss /= i
    print(f" validation loss {final_val_loss}")
    return model

def test_model(data, model):      
    model.eval()
    with torch.no_grad():
        outputs = model(data)
    return outputs


# In[30]:



z_Score_normalize = True

if z_Score_normalize == True:
    train = z_Score_norm(train)


train_tensor = torch.tensor(train[train_columns].to_numpy()).double()
target_tensor = torch.tensor(target[target_columns].to_numpy()).double() #,dtype=torch.double
test_tensor = torch.tensor(test[train_columns].to_numpy()).double()

# hyperparameters
batch_size  = 1024
hidden_size = 1024
learning_rate = 1e-3
weight_decay = 1e-5
num_epochs = 50

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device, {DEVICE}")
train_tensor = train_tensor.to(DEVICE)
target_tensor = target_tensor.to(DEVICE)
test_tensor = test_tensor.to(DEVICE)
print(train_tensor.device,target_tensor.device)



dataset = TensorDataset(train_tensor.double(),target_tensor)
train_set, val_set = torch.utils.data.random_split(dataset, [20000,train.shape[0]-20000])
train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_data = DataLoader(val_set, batch_size=batch_size, shuffle=True)


model = Model(len(train_columns),len(target_columns),hidden_size,dtype)
model.to(DEVICE).to(dtype)
# model.double()



model = train_model(train_data,val_data,model,learning_rate,weight_decay)
# predictions = test_model(test_tensor,model)


# In[ ]:


validation loss 0.004120627779762144


# In[ ]:


submit = pd.DataFrame(predictions.numpy())
submit.columns = list(target_columns)
x_test_id = test['sig_id']
submit = pd.concat([x_test_id,submit],axis=1)


# In[ ]:


submit.to_csv('submission.csv', index=False)


# In[ ]:


# from sklearn.metrics import log_loss

# xx,yy = train_set[:]
# y_pred = test_model(xx,model)
# ll = log_loss(yy,y_pred)
# print(ll)


# In[ ]:


train_drug['sig_id']=='id_008a986b7'


# In[ ]:


train[train_drug['sig_id']=='id_008a986b7']['cp_type']=='ctl_vehicle'


# In[21]:


t2 = train_tensor.to(DEVICE)
print(t2.device)


# In[ ]:


train['cp_type']=='ctl_vehicle'

