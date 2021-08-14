#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
project_name = "reco-tut-sor"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)


# In[3]:


if not os.path.exists(project_path):
    get_ipython().system(u'cp /content/drive/MyDrive/mykeys.py /content')
    import mykeys
    get_ipython().system(u'rm /content/mykeys.py')
    path = "/content/" + project_name; 
    get_ipython().system(u'mkdir "{path}"')
    get_ipython().magic(u'cd "{path}"')
    import sys; sys.path.append(path)
    get_ipython().system(u'git config --global user.email "recotut@recohut.com"')
    get_ipython().system(u'git config --global user.name  "reco-tut"')
    get_ipython().system(u'git init')
    get_ipython().system(u'git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git')
    get_ipython().system(u'git pull origin "{branch}"')
    get_ipython().system(u'git checkout main')
else:
    get_ipython().magic(u'cd "{project_path}"')


# In[50]:


get_ipython().system(u'git status')


# In[51]:


get_ipython().system(u'git add . && git commit -m \'commit\' && git push origin "{branch}"')


# ---

# In this notebook we will be building XGB model and check if the reccomendation engine can be improved by using other algorithms

# In[23]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix,  accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score
import itertools

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb


# ## Creating alternative model
# 
# In this step dataset preprocessed in previous step is loaded and simple baseline model is tested.
# 
# Each line in a dataset contains data about one user and his final action on the offer. 
# Either offer has been ignored, viewed or completed (offer proved to be interesting to a customer).

# In[5]:


df = pd.read_csv('./data/silver/userdata.csv')


# In[6]:


df.head()


# In[7]:


print("Dataset contains %s actions" % len(df))


# ### Let's plot the actions for one user
# 
# From the output can be seen that user completed an offer `0b1e...` and viewed `ae26...`. Offer `2906..` had been ignored twice.

# In[8]:


df[df.id == 'e12aeaf2d47d42479ea1c4ac3d8286c6' ]


# ### Preparing data for training
# Let's create user-offer matrix by encoding each id into categorical value.

# Recommendation matrix is very similar to embeddings. So we will leverage this and will train embedding along the model.

# ### Create additional user and offer details tensors

# In[9]:


offer_specs = ['difficulty', 'duration', 'reward', 'web',
       'mobile', 'social', 'bogo', 'discount', 'informational']
user_specs = ['age', 'became_member_on', 'gender', 'income', 'memberdays']


# In[10]:


N_train = int(0.8 * len(df['event']))
N_test = 1000

train_df = df[:N_train]
test_df = df[N_train:]
print(len(train_df))
print(len(test_df))


# In[11]:


def random_forest(train_data, train_true, test_data, test_true):
    #hyper-paramater tuning
    values = [25, 50, 100, 200]
    clf = RandomForestClassifier(n_jobs = -1)
    hyper_parameter = {"n_estimators": values}
    best_parameter = GridSearchCV(clf, hyper_parameter, scoring = "neg_mean_absolute_error", cv = 3)
    best_parameter.fit(train_data, train_true)
    estimators = best_parameter.best_params_["n_estimators"]
    print("Best RF parameter is: ", estimators)
    #applying random forest with best hyper-parameter
    clf = RandomForestClassifier(n_estimators = estimators, n_jobs = -1)
    clf.fit(train_data, train_true)
    #train_pred = clf.predict(train_data)
    return clf

def xgboost_model(train_data, train_true, test_data, test_true):
    #hyper-parameter tuning
    hyper_parameter = {"max_depth":[6, 8, 10, 16], "n_estimators":[60, 80, 100, 120]}
    clf = xgb.XGBClassifier()
    best_parameter = GridSearchCV(clf, hyper_parameter, scoring = "neg_mean_absolute_error", cv = 3)
    best_parameter.fit(train_data, train_true)
    estimators = best_parameter.best_params_["n_estimators"]
    depth = best_parameter.best_params_["max_depth"]
    print("Best XGB parameter is %s estimators and depth %s: " % (estimators, depth))
    clf = xgb.XGBClassifier(max_depth = depth, n_estimators = estimators)
    clf.fit(train_data, train_true)
    #train_pred = clf.predict(train_data)
    return clf


# In[12]:


pred_rf = random_forest(train_df[user_specs + offer_specs], 
                       train_df['event'].values.ravel(), 
                       test_df[user_specs + offer_specs],
                       test_df['event'].values.ravel())

pred_xgb = xgboost_model(train_df[user_specs + offer_specs], 
                       train_df['event'].values.ravel(), 
                       test_df[user_specs + offer_specs],
                        test_df['event'].values.ravel())

# error_table_regressions = pd.DataFrame(columns = ["Model", "TrainMAPE(%)", "TrainMSE", "TestMAPE(%)", "TestMSE"])
# error_table_regressions = error_table_regressions.append(pd.DataFrame([["XGBoost Regressor", trainMAPE_xgb*100, trainMSE_xgb, testMAPE_xgb*100, testMSE_xgb]], columns = ["Model", "TrainMAPE(%)", "TrainMSE", "TestMAPE(%)", "TestMSE"]))
# error_table_regressions = error_table_regressions.append(pd.DataFrame([["Random Forest Regression", trainMAPE_rf*100, trainMSE_rf, testMAPE_rf*100, testMSE_rf]], columns = ["Model", "TrainMAPE(%)", "TrainMSE", "TestMAPE(%)", "TestMSE"]))

# error_table_regressions.reset_index(drop = True, inplace = True)


# In[14]:


def random_forest(train_data, train_true, test_data, test_true):
    clf = RandomForestClassifier(n_estimators = 60, n_jobs = -1)
    clf.fit(train_data, train_true)
    #train_pred = clf.predict(train_data)
    return clf

def xgboost_model(train_data, train_true, test_data, test_true):
    #hyper-parameter tuning
    clf = xgb.XGBClassifier(max_depth = 16, n_estimators = 6)
    clf.fit(train_data, train_true)
    #train_pred = clf.predict(train_data)
    return clf


# In[15]:


pred_rf = random_forest(train_df[user_specs + offer_specs], 
                       train_df['event'].values.ravel(), 
                       test_df[user_specs + offer_specs],
                       test_df['event'].values.ravel())

pred_xgb = xgboost_model(train_df[user_specs + offer_specs], 
                       train_df['event'].values.ravel(), 
                       test_df[user_specs + offer_specs],
                        test_df['event'].values.ravel())

# error_table_regressions = pd.DataFrame(columns = ["Model", "TrainMAPE(%)", "TrainMSE", "TestMAPE(%)", "TestMSE"])
# error_table_regressions = error_table_regressions.append(pd.DataFrame([["XGBoost Regressor", trainMAPE_xgb*100, trainMSE_xgb, testMAPE_xgb*100, testMSE_xgb]], columns = ["Model", "TrainMAPE(%)", "TrainMSE", "TestMAPE(%)", "TestMSE"]))
# error_table_regressions = error_table_regressions.append(pd.DataFrame([["Random Forest Regression", trainMAPE_rf*100, trainMSE_rf, testMAPE_rf*100, testMSE_rf]], columns = ["Model", "TrainMAPE(%)", "TrainMSE", "TestMAPE(%)", "TestMSE"]))

# error_table_regressions.reset_index(drop = True, inplace = True)


# In[16]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save=False,
                          figname='cm.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save:
        plt.savefig(figname, dpi=fig.dpi)
    plt.show()


# In[17]:


pred1 = pred_rf.predict(test_df[0:1000][user_specs + offer_specs])
test_y = test_df[0:1000]['event'].values.ravel()
#print(pred1)
#print(test_y)
cm = confusion_matrix(test_y, pred1)
classes = [0,1,2]
plot_confusion_matrix(cm, classes, save=True, figname='./outputs/Recommendation-cm.png')


# In[18]:


pred2 = pred_xgb.predict(test_df[0:1000][user_specs + offer_specs])
test_y = test_df[0:1000]['event'].values.ravel()
#print(pred2)
#print(test_y)
cm = confusion_matrix(test_y, pred2)
classes = [0,1,2]
plot_confusion_matrix(cm, classes, save=True, figname='./outputs/RecommendationXGB-cm.png')


# In[19]:


print("Accuracy for RF model: " + str(100*accuracy_score(test_y, pred1))+ "%" )
print("Accuracy for XGB model: " + str(100*accuracy_score(test_y, pred2))+ "%" )


# In[20]:


print("F1 score for RF model: " + str(f1_score(test_y, pred1, average='weighted')))
print("Recall score for RF model: " + str(recall_score(test_y, pred1, average='weighted')))
print("Precision score for RF model: " + str(precision_score(test_y, pred1, average='weighted')))

print("")
print("F1 score for XGB model: " + str(f1_score(test_y, pred2, average='weighted')) )
print("Recall score for XGB model: " + str(recall_score(test_y, pred2, average='weighted')) )
print("Precision score for XGB model: " + str(precision_score(test_y, pred2, average='weighted')) )


# Results seem to be promising.
# Let's try to improve them even more, and simplify data as from the correlation matrix it can be noticed that model has difficulties to differentiate if user will view an offer or even respond to it.
# This can be due to the fact that responding to an offer implies that user had definitely viewed an offer.

# ## Approach 2. Remove outlier fields

# In[26]:


df = pd.read_csv('./data/silver/userdata.csv')


# In[27]:


df['member_days'] = pd.to_datetime(df['became_member_on'], format="%Y%m%d")
df['member_days'] = df['member_days'] - df['member_days'].min()
df['member_days'] = df['member_days'].apply(lambda x: int(x.days))


# Let's check once again the correlation between gender and event response.
# We are interested in X and O genders. Where X is the customers with anonymized data.

# In[28]:


df[df.gender == 0]['event'].plot.hist()#.count_values()


# In[29]:


df[df.gender == 1]['event'].plot.hist()#.count_values()


# In[30]:


df[df.gender == 2]['event'].plot.hist()#.count_values()


# In[31]:


df[df.gender == 3]['event'].plot.hist()#.count_values()


# In[32]:


df[df.income == 0]['event'].plot.hist()#.count_values()


# Now we test the model performance with removing rows where user with age and income as None
# They seem to view offer but rarely respond to it.

# In[33]:


# We remove them by index as it seems to be the easiest way
indexes_to_drop = list(df[df.gender == 0].index) + list(df[df.income == 0].index)
df = df.drop(df.index[indexes_to_drop]).reset_index()


# In[34]:


df = df.reset_index()


# In[36]:


df['became_member_date'] = pd.to_datetime(df['became_member_on'], format="%Y%m%d")
df[df['member_days'] == 10]


# Let's encode `event` field to be only binary value, with event ignored as 0, and offer completed - as 1.

# In[37]:


df['event'] = df['event'].map({0:0, 1:0, 2:1})


# In[38]:


offer_specs = ['difficulty', 'duration', 'reward', 'web', 'email',
       'mobile', 'social', 'bogo', 'discount', 'informational']
user_specs = ['age', 'member_days', 'gender', 'income']


# In[39]:


N_train = int(0.8 * len(df['event']))
N_test = 1000

train_df = df[:N_train]
test_df = df[N_train:]
print(len(train_df))
print(len(test_df))


# In[40]:


pred_rf = random_forest(train_df[user_specs + offer_specs], 
                       train_df['event'].values.ravel(), 
                       test_df[user_specs + offer_specs],
                       test_df['event'].values.ravel())

pred_xgb = xgboost_model(train_df[user_specs + offer_specs], 
                       train_df['event'].values.ravel(), 
                       test_df[user_specs + offer_specs],
                        test_df['event'].values.ravel())


# In[41]:


pred1 = pred_rf.predict(test_df[user_specs + offer_specs])
test_y = test_df['event'].values.ravel()
print(pred1)
print(test_y)

print("Accuracy for RF model: " + str(100*accuracy_score(test_y, pred1))+ "%" )
cm = confusion_matrix(test_y, pred1)
classes = [0,1,2]
plot_confusion_matrix(cm, classes)


# In[42]:


pred2 = pred_xgb.predict(test_df[user_specs + offer_specs])
test_y = test_df['event'].values.ravel()
print(pred2)
print(test_y)

print("Accuracy for XGB model: " + str(100*accuracy_score(test_y, pred2))+ "%" )
cm = confusion_matrix(test_y, pred2)
classes = [0,1,2]
plot_confusion_matrix(cm, classes)


# It seem that results are the same.
# Let's try the model with encoding now 
# an `event` field to be only binary value, with event ignored as 0, and offer completed - as 1.

# ## Approach 3. Building Performance optimized model

# In[45]:


df = pd.read_csv('./data/silver/userdata.csv')

df['member_days'] = pd.to_datetime(df['became_member_on'], format="%Y%m%d")
df['member_days'] = df['member_days'] - df['member_days'].min()
df['member_days'] = df['member_days'].apply(lambda x: int(x.days))

df['event'] = df['event'].map({0:0, 1:1, 2:1})

df = df.reset_index()

offer_specs = ['difficulty', 'duration', 'reward', 'web', 'email',
       'mobile', 'social', 'bogo', 'discount', 'informational']
user_specs = ['age', 'member_days', 'gender', 'income']

N_train = int(0.8 * len(df['event']))
N_test = 1000

train_df = df[:N_train]
test_df = df[N_train:]
print(len(train_df))
print(len(test_df))

def random_forest(train_data, train_true, test_data, test_true):
   
    clf = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
    clf.fit(train_data, train_true)
    #train_pred = clf.predict(train_data)
    
    return clf

def xgboost_model(train_data, train_true, test_data, test_true):
    
    clf = xgb.XGBClassifier(max_depth = 16, n_estimators = 60)
    clf.fit(train_data, train_true)
    #train_pred = clf.predict(train_data)
    
    return clf


# In[46]:


pred_rf = random_forest(train_df[user_specs + offer_specs], 
                       train_df['event'].values.ravel(), 
                       test_df[user_specs + offer_specs],
                       test_df['event'].values.ravel())

pred_xgb = xgboost_model(train_df[user_specs + offer_specs], 
                       train_df['event'].values.ravel(), 
                       test_df[user_specs + offer_specs],
                        test_df['event'].values.ravel())


# In[47]:


pred1 = pred_rf.predict(test_df[0:1000][user_specs + offer_specs])
test_y = test_df[0:1000]['event'].values.ravel()
#print(pred1)
#print(test_y)

print("Accuracy for RF model: " + str(100*accuracy_score(test_y, pred1))+ "%" )
cm = confusion_matrix(test_y, pred1)
classes = [0,1,2]
plot_confusion_matrix(cm, classes, save=True, figname='./outputs/RF-model-cm.png')


# In[48]:


pred2 = pred_xgb.predict(test_df[0:1000][user_specs + offer_specs])
test_y = test_df[0:1000]['event'].values.ravel()
#print(pred2)
#print(test_y)

print("Accuracy for XGB model: " + str(100*accuracy_score(test_y, pred2))+ "%" )
cm = confusion_matrix(test_y, pred2)
classes = [0,1,2]
plot_confusion_matrix(cm, classes, save=True, figname='./outputs/XGB-model-cm.png')


# This looks like a significant improve that can be used in production to save costs and send offers to those users who are going to be interested in companies offers without ignoring them.

# In[49]:


print("F1 score for RF model: " + str(f1_score(test_y, pred1, average='weighted')))
print("Recall score for RF model: " + str(recall_score(test_y, pred1, average='weighted')))
print("Precision score for RF model: " + str(precision_score(test_y, pred1, average='weighted')))

print("")
print("F1 score for XGB model: " + str(f1_score(test_y, pred2, average='weighted')) )
print("Recall score for XGB model: " + str(recall_score(test_y, pred2, average='weighted')) )
print("Precision score for XGB model: " + str(precision_score(test_y, pred2, average='weighted')) )


# This proves to be a very good model for ad hoc predictions and predictions on subsections of customer by regions or cities.
