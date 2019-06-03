#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score


# In[3]:


train = pd.read_csv('train.csv', encoding = "ISO-8859-1")
test = pd.read_csv('test.csv')
addresses = pd.read_csv('addresses.csv')
latlons = pd.read_csv('latlons.csv')


# In[4]:


test.head()


# In[5]:


len(train)


# In[6]:


train = train[np.isfinite(train['compliance'])]


# In[7]:


len(train)


# In[8]:


train.columns


# In[9]:


train = train[train['country'] == 'USA']
test = test[test['country'] == 'USA']


# In[10]:


len(train)


# In[11]:


addresses.head()


# In[12]:


latlons.head()


# In[13]:


train = pd.merge(train, pd.merge(addresses, latlons, on = 'address'), on = 'ticket_id')


# In[14]:


test = pd.merge(test, pd.merge(addresses, latlons, on = 'address'), on = 'ticket_id')


# In[15]:


train.head()


# In[16]:


train


# In[17]:


train.drop(['agency_name', 'inspector_name', 'violator_name', 'non_us_str_code', 'violation_description', 'grafitti_status', 'state_fee', 'admin_fee', 'ticket_issued_date', 'hearing_date','payment_amount', 'balance_due', 'payment_date', 'payment_status','collection_status', 'compliance_detail', 'violation_zip_code', 'country', 'address', 'violation_street_number','violation_street_name', 'mailing_address_str_number', 'mailing_address_str_name', 'city', 'state', 'zip_code', 'address'], axis=1, inplace=True)


# In[18]:


train.head()


# In[19]:


label_encoder = LabelEncoder()


# In[20]:


label_encoder.fit(train['disposition'].append(test['disposition'], ignore_index = True))


# In[21]:


label_encoder.classes_


# In[22]:


train['disposition'] = label_encoder.transform(train['disposition'])


# In[23]:


train['disposition']


# In[24]:


test['disposition'] = label_encoder.transform(test['disposition'])


# In[25]:


test['disposition']


# In[26]:


label_encoder = LabelEncoder()


# In[28]:


label_encoder.fit(train['violation_code'].append(test['violation_code'], ignore_index = True))


# In[29]:


train['violation_code'] = label_encoder.transform(train['violation_code'])


# In[30]:


test['violation_code'] = label_encoder.transform(test['violation_code'])


# In[31]:


test['violation_code']


# In[32]:


label_encoder.classes_


# In[33]:


train['lat'] = train['lat'].fillna(train['lat'].mean())


# In[35]:


train['lon'] = train['lon'].fillna(train['lon'].mean())


# In[36]:


test['lat'] = test['lat'].fillna(test['lat'].mean())


# In[37]:


test['lon'] = test['lon'].fillna(test['lon'].mean())


# In[40]:


train_columns = list(train.columns.values)


# In[41]:


train_columns.remove('compliance')


# In[42]:


test = test[train_columns]


# In[43]:


X = train.ix[:, train.columns != 'compliance']


# In[46]:


y = train['compliance']


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[48]:


regr_rf = RandomForestRegressor()


# In[49]:


grid_values = {'n_estimators': [10, 80], 'max_depth': [None, 20]}


# In[53]:


grid_clf_auc = GridSearchCV(regr_rf, param_grid = grid_values, scoring = 'roc_auc')
grid_clf_auc.fit(X_train, y_train)


# In[51]:


print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)


# In[52]:


print('Grid best score (AUC): ', grid_clf_auc.best_score_)


# In[54]:


pd.DataFrame(grid_clf_auc.predict(test), test.ticket_id)


# In[ ]:




