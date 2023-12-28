#!/usr/bin/env python
# coding: utf-8

# # if a stand-up comedy will receive above or below average IMDb rating
# 
# 1) Train weak learners: Random Forrest, Stochastic Gradient Descent.
# 
# 2) Perform a grid search to find optimal parameters for an XGBoost classifier.
# 
# 3) Put all three models into an ensemble.

# In[1]:


import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')


# In[4]:


df=pd.read_csv("frame4.csv")


# In[6]:


df


# ### One-hot features for cluster assignments

# In[7]:


cluster_LDA_dummies = pd.get_dummies(df['cluster_LDA'])
LDA_columns = [str(column) + '_LDA' for column in cluster_LDA_dummies.columns]
cluster_LDA_dummies.columns = LDA_columns

cluster_tfidf_dummies = pd.get_dummies(df['cluster_tfidf'])
tfidf_columns = [str(column) + '_tfidf' for column in cluster_tfidf_dummies.columns]
cluster_tfidf_dummies.columns = tfidf_columns

cluster_df = pd.merge(cluster_LDA_dummies, cluster_tfidf_dummies, right_index=True, left_index=True)
cluster_df.head()


# In[8]:


df = pd.merge(df, cluster_df, right_index=True, left_index=True)
df.columns


# ### Split data into training and testing sets and train models.
# 
# - Train Random Forest model
# 
# - Train SGD model
# 
# - Perform grid search and train XGB model
# 
# - Create and ensemble of three classifiers

# ## Only LDA Topic assignments to train the model

# In[9]:


X = np.array(df[['Culture', 'UK', 'Crimes', 'Situational', 'Immigrants', 'Relationships', 'Politics']].loc[df.rating > 0])
y = np.array(df.rating_type.loc[df.rating > 0])
print(X.shape)
print(y.shape)


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)


# In[11]:


# Random Forrest
rf = RandomForestClassifier(n_estimators=101).fit(X_train, y_train)
print(f'RF score: {rf.score(X_test, y_test)}')


# In[12]:


# SGD
sgd = linear_model.SGDClassifier(loss='modified_huber').fit(X_train, y_train)
print(f'SGD score: {sgd.score(X_test, y_test)}')


# In[13]:


xgb = XGBClassifier()
parameters = {
     "eta"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
     "min_child_weight" : [ 1, 3, 5, 7 ],
     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
     }

grid = GridSearchCV(xgb,
                    parameters, n_jobs=4,
                    scoring="neg_log_loss",
                    cv=3)

grid.fit(X_train, y_train)


# In[14]:


best_xgb = grid.best_estimator_.fit(X_train, y_train)
print(f'Best params: {grid.best_params_}')
print(f'Best XGB score: {best_xgb.score(X_test, y_test)}')


# In[16]:


# Ensemble
estimators = [('rf', rf), ('sgd', sgd), ('xgb', best_xgb)]

ensemble = VotingClassifier(estimators, voting='soft')
ensemble.fit(X_train, y_train)
print('Voting Classifier, Ensemble Acc: {}'.format(ensemble.score(X_test, y_test)))


# ## Only Cluster assignments to train the model

# In[17]:


X = np.array(df[['0_LDA', '1_LDA', '2_LDA', '3_LDA',
       '4_LDA', '5_LDA', '6_LDA', '0_tfidf', '1_tfidf', '2_tfidf', '3_tfidf',
       '4_tfidf', '5_tfidf', '6_tfidf']].loc[df.rating > 0])
y = np.array(df.rating_type.loc[df.rating > 0])
print(X.shape)
print(y.shape)


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)


# In[18]:


# Random Forrest
rf = RandomForestClassifier(n_estimators=101).fit(X_train, y_train)
print(f'RF score: {rf.score(X_test, y_test)}')


# In[19]:


# SGD
sgd = linear_model.SGDClassifier(loss='modified_huber').fit(X_train, y_train)
print(f'SGD score: {sgd.score(X_test, y_test)}')


# In[20]:


# XGBoosting
xgb = XGBClassifier().fit(X_train, y_train)
print(f'XGB score: {xgb.score(X_test, y_test)}')


# In[21]:


# Ensemble
estimators = [('rf', rf), ('sgd', sgd), ('xgb', xgb)]

ensemble = VotingClassifier(estimators, voting='soft')
ensemble.fit(X_train, y_train)
print('Voting Classifier, Ensemble Acc: {}'.format(ensemble.score(X_test, y_test)))


# ## Both cluster assignments and LDA probabilities

# In[22]:


X = np.array(df[['Culture', 'UK', 'Crimes', 'Situational', 'Immigrants', 'Relationships', 'Politics', '0_LDA', '1_LDA', '2_LDA', '3_LDA',
                 '4_LDA', '5_LDA', '6_LDA', '0_tfidf', '1_tfidf', '2_tfidf', '3_tfidf',
                 '4_tfidf', '5_tfidf', '6_tfidf']].loc[df.rating > 0])
y = np.array(df.rating_type.loc[df.rating > 0])
print(X.shape)
print(y.shape)


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)


# In[25]:


# Random Forrest
rf = RandomForestClassifier(n_estimators=101).fit(X_train, y_train)
print(f'RF score: {rf.score(X_test, y_test)}')


# In[26]:


# SGD
sgd = linear_model.SGDClassifier(loss='modified_huber').fit(X_train, y_train)
print(f'SGD score: {sgd.score(X_test, y_test)}')


# In[27]:


xgb = XGBClassifier()
parameters = {
     "eta"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
     "min_child_weight" : [ 1, 3, 5, 7 ],
     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
     }

grid = GridSearchCV(xgb,
                    parameters, n_jobs=4,
                    scoring="neg_log_loss",
                    cv=3)

grid.fit(X_train, y_train)


# In[28]:


best_xgb = grid.best_estimator_.fit(X_train, y_train)
print(f'Best params: {grid.best_params_}')
print(f'Best XGB score: {best_xgb.score(X_test, y_test)}')


# In[29]:


# Ensemble
estimators = [('rf', rf), ('sgd', sgd), ('xgb', best_xgb)]

ensemble = VotingClassifier(estimators, voting='soft')
ensemble.fit(X_train, y_train)
print('Voting Classifier, Ensemble Acc: {}'.format(ensemble.score(X_test, y_test)))


# ### The Random Forest performed the best at 0.68 accuracy when taking only cluster assignments.

# In[ ]:




