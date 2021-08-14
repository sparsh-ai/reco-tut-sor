#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
project_name = "reco-tut-sor"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)


# In[ ]:


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


# In[103]:


get_ipython().system(u'git status')


# In[104]:


get_ipython().system(u'git add . && git commit -m \'commit\' && git push origin "{branch}"')


# ---

# This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.
# 
# Not all users receive the same offer, and that is the challenge to solve with this data set.
# 
# The task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.
# 
# Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.
# 
# The provided transactional data shows user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer.
# 
# Let's keep in mind as well that someone using the app might make a purchase through the app without having received an offer or seen an offer.
# 
# To give an example, a user could receive a discount offer buy 10 dollars get 2 off on Monday. The offer is valid for 10 days from receipt. If the customer accumulates at least 10 dollars in purchases during the validity period, the customer completes the offer.
# 
# However, there are a few things to watch out for in this data set. Customers do not opt into the offers that they receive; in other words, a user can receive an offer, never actually view the offer, and still complete the offer. For example, a user might receive the "buy 10 dollars get 2 dollars off offer", but the user never opens the offer during the 10 day validity period. The customer spends 15 dollars during those ten days. There will be an offer completion record in the data set; however, the customer was not influenced by the offer because the customer never viewed the offer.
# 
# You'll also want to take into account that some demographic groups will make purchases even if they don't receive an offer. From a business perspective, if a customer is going to make a 10 dollar purchase without an offer anyway, you wouldn't want to send a buy 10 dollars get 2 dollars off offer. You'll want to try to assess what a certain demographic group will buy when not receiving any offers.

# ## Dataset
# 
# The data is contained in three files:
# 
# * portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
# * profile.json - demographic data for each customer
# * transcript.json - records for transactions, offers received, offers viewed, and offers completed
# 
# Here is the schema and explanation of each variable in the files:
# 
# **portfolio.json**
# * id (string) - offer id
# * offer_type (string) - type of offer ie BOGO, discount, informational
# * difficulty (int) - minimum required spend to complete an offer
# * reward (int) - reward given for completing an offer
# * duration (int) - time for offer to be open, in days
# * channels (list of strings)
# 
# **profile.json**
# * age (int) - age of the customer 
# * became_member_on (int) - date when customer created an app account
# * gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
# * id (str) - customer id
# * income (float) - customer's income
# 
# **transcript.json**
# * event (str) - record description (ie transaction, offer received, offer viewed, etc.)
# * person (str) - customer id
# * time (int) - time in hours since start of test. The data begins at time t=0
# * value - (dict of strings) - either an offer id or transaction amount depending on the record

# In[ ]:


import datetime
import pandas as pd
import numpy as np
import math
import json
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

get_ipython().magic(u'matplotlib inline')


# In[60]:


# read in the json files
portfolio = pd.read_json('./data/bronze/portfolio.json', orient='records', lines=True)
profile = pd.read_json('./data/bronze/profile.json', orient='records', lines=True)
transcript = pd.read_json('./data/bronze/transcript.json', orient='records', lines=True)


# ## Portfolio

# | attribute | description |
# | --------- | ----------- |
# | id | offer id |
# | offer_type | type of offer ie BOGO, discount, informational |
# | difficulty | minimum required spend to complete an offer |
# | reward | reward given for completing an offer |
# | duration | time for offer to be open, in days |
# | channels | email, web, mobile |

# In[61]:


portfolio


# In[62]:


portfolio.info()


# In[63]:


portfolio.describe().round(1)


# In[64]:


fig, ax = plt.subplots(figsize=(12,7))
portfolio.hist(ax=ax)
plt.show()


# In[65]:


portfolio.describe(include='O')


# In[66]:


portfolio.channels.astype('str').value_counts().plot(kind='barh');


# In[67]:


portfolio.offer_type.value_counts().plot(kind='barh');


# ## Transcript

# In[68]:


transcript.head()


# In[69]:


transcript.info()


# In[70]:


transcript.describe().round(1).T


# In[71]:


transcript.describe(include='O')


# In[72]:


transcript.event.astype('str').value_counts().plot(kind='barh');


# ## Profile

# In[73]:


profile.head()


# In[74]:


profile.info()


# In[75]:


profile.describe().round(1)


# In[76]:


fig, ax = plt.subplots(figsize=(12,7))
profile.hist(ax=ax)
plt.show()


# In[77]:


profile.describe(include='O')


# In[78]:


profile.gender.astype('str').value_counts(dropna=False).plot(kind='barh');


# ## Cleaning the data and Feature Engineering
# 

# In[79]:


group_income = profile.groupby(['income', 'gender']).size().reset_index()
group_income.columns = ['income', 'gender', 'count']

sns.catplot(x="income", y="count", hue="gender", data=group_income,
                  kind="bar", palette="muted", height=5, aspect=12/5)
plt.xlabel('Income per year')
plt.ylabel('Count')
plt.title('Age/Income Distribution')
plt.savefig('./extras/images/income-age-dist-binned.png', dpi=fig.dpi)


# In[80]:


portfolio['web'] = portfolio['channels'].apply(lambda x: 1 if 'web' in x else 0)
portfolio['email'] = portfolio['channels'].apply(lambda x: 1 if 'email' in x else 0)
portfolio['mobile'] = portfolio['channels'].apply(lambda x: 1 if 'mobile' in x else 0)
portfolio['social'] = portfolio['channels'].apply(lambda x: 1 if 'social' in x else 0)
    
# apply one hot encoding to offer_type column
offer_type = pd.get_dummies(portfolio['offer_type'])

# drop the channels and offer_type column
portfolio.drop(['channels', 'offer_type'], axis=1, inplace=True)

# combine the portfolio and offer_type dataframe to form a cleaned dataframe
portfolio = pd.concat([portfolio, offer_type], axis=1, sort=False)


# In[81]:


profile['memberdays'] = datetime.datetime.today().date() - pd.to_datetime(profile['became_member_on'], format='%Y%m%d').dt.date
profile['memberdays'] = profile['memberdays'].dt.days
profile['income'] = profile['income'].fillna(0)

profile['gender'] = profile['gender'].fillna('X')
profile['gender'] = profile['gender'].map({'X':0,'O':1, 'M':2, 'F':3})
income_bins = [0, 20000, 35000, 50000, 60000, 70000, 90000, 100000, np.inf]
labels = [0,1,2,3,4,5,6,7]
profile['income'] = pd.cut(profile['income'], bins = income_bins, labels= labels, include_lowest=True)


# In[82]:


# Let's plot the sama data and see if this provide us with better insights

group_income = profile.groupby(['income', 'gender']).size().reset_index()
group_income.columns = ['income', 'gender', 'count']

sns.catplot(x="income", y="count", hue="gender", data=group_income,
                  kind="bar", palette="muted", height=5, aspect=12/5)
plt.xlabel('Income per year')
plt.ylabel('Count')
plt.title('Age/Income Distribution')
plt.savefig('./extras/images/income-age-dist-binned.png', dpi=fig.dpi)


# ## Joining the data

# In[87]:


transcript = transcript[transcript.person != None]
# extract ids for each offer
transcript['offer_id'] = transcript[transcript.event != 'transaction']['value'].apply(lambda x: 
                                                             dict(x).get('offer id') 
                                                             if dict(x).get('offer id') is not None 
                                                             else dict(x).get('offer_id') )

# transaction offers does not have offer id, so we filter them out next
joined_df = pd.merge(profile, transcript[transcript.event != 'transaction'], how='left', left_on=['id'], right_on=['person'])
joined_df['event'] = joined_df['event'].map({'offer received': 0, 'offer viewed': 1, 'offer completed': 2})

# rename column for ease of joining of dataframes
portfolio.rename({'id':'offer_id'}, inplace=True, axis=1)

# now all data can be joined together
df = pd.merge(joined_df, portfolio, how='inner', left_on=['offer_id'], right_on=['offer_id'])
df = df.drop(['person', 'value'], axis=1)

df.head()


# ## Exploring correlations
# 
# Correlation is used to find which values are closely related with each other.
# Now let's describe how values are correlated with each ther. For simplicity - the size of the output dot will define the correlation (the bigger - the closer).

# In[88]:


#!mkdir images
def heatmap(x, y, size, figsize=(18,15), fig_name='temp.png'):
    fig, ax = plt.subplots(figsize=figsize)
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size * size_scale, # Vector of square sizes, proportional to size parameter
        marker='s' # Use square as scatterplot marker
    )
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    fig.savefig(fig_name, dpi=fig.dpi)
    
offer_specs = ['difficulty', 'duration', 'reward', 'web',
       'email', 'mobile', 'social', 'bogo', 'discount', 'informational']
user_specs = ['age', 'became_member_on', 'gender', 'income', 'memberdays']

corr = df[offer_specs + user_specs + ['event']].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'event']
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['event'].abs(),
    fig_name='./extras/images/heatmap-general.png'
)


# Correlation between features seems to be quite weak. However it can be noted that `bogo` is strongly related to `discount` and `reward` fields, while `mobile` channel is correlated with `difficulty` field. Which is quite expected.
# 
# Now let's see more closely into columns of our interest and define if this should be cleaned or changed.

# In[89]:


corr = df[['income', 'gender','event']].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs(),
    figsize=(4,4),
    fig_name='./extras/images/heatmap-event.png'
)


# ## Building Recommendation matrix

# At the moment data for each user has entries for each offer if it was received, viewed and responded to it.
# To be able to give valid recommendations we leave only last user action on each offer (either viewed, responded or ignored).

# In[ ]:


df[(df.id == '68be06ca386d4c31939f3a4f0e3dd783') & (df.offer_id == '2906b810c7d4411798c6938adc9daaa5')]


# In[97]:


users = df['id'].unique()
offers = df['offer_id'].unique()
recommendation_df = pd.DataFrame(columns=df.columns)

recommendation_df.head()


# In[ ]:


print("Number of known users: ", len(users))
print("Number of created offers: ", len(offers))


# In[98]:


for i, offer in enumerate(offers):
    for j, user in enumerate(users):
        offer_id_actions = df[(df.id == user) & (df.offer_id == offer)]
        # log progress 
        if j % 5000 == 0:
            print('Processing offer %s for user with index: %s' % (i, j))        
        if len(offer_id_actions) > 1:
            # user viewed or resonded to offer
            if offer_id_actions[offer_id_actions.event == 2]['event'].empty == False:
                # user has not completed an offer
                recommendation_df = recommendation_df.append(offer_id_actions[offer_id_actions.event == 2])
            elif offer_id_actions[offer_id_actions.event == 1]['event'].empty == False:
                # user only viewed offer
                recommendation_df = recommendation_df.append(offer_id_actions[offer_id_actions.event == 1])
            else:
                # Offer could be de received multiple times but ignored
                #print("Filter length", len())
                #print("No event were found in filtered data\n:", offer_id_actions)
                recommendation_df = recommendation_df.append(offer_id_actions[offer_id_actions.event == 0])
        else:
            # offer has been ignored
            recommendation_df = recommendation_df.append(offer_id_actions[offer_id_actions.event == 0])


# In[99]:


recommendation_df.head()


# In[100]:


recommendation_df['event'][10000:50000].map({0:'offer received', 1: 'offer viewed', 2: 'offer completed'}).value_counts().plot.pie(figsize=(7, 7), 
                                       title="Event Pie Chart", 
                                       autopct='%1.1f%%', 
                                       legend=True)


# In[90]:


gr = df.groupby(['id','offer_id'])
user_actions = pd.concat([gr.tail(1)]).reset_index(drop=True)
user_actions.head()


# In[91]:


user_actions[user_actions.id == 'e12aeaf2d47d42479ea1c4ac3d8286c6']


# In[92]:


user_actions['event'][0:1000].map({0:'offer received', 1: 'offer viewed', 2: 'offer completed'}).value_counts().plot.pie(figsize=(7, 7), 
                                       title="Event Pie Chart", 
                                       autopct='%1.1f%%', 
                                       legend=True)


# Final users/offers datasets look pretty good, however we still not able to extract some actions perfomed by users, especially with filtering duplicates. This might be caused by the fact when offer was received twice.
# 
# Let's filter them and explore once more.

# In[93]:


user_actions.drop_duplicates(subset=['id', 'offer_id'], keep=False)

user_actions[user_actions.id == 'e12aeaf2d47d42479ea1c4ac3d8286c6' ]


# In[94]:


user_actions['event'][0:1000].map({0:'offer received', 1: 'offer viewed', 2: 'offer completed'}).value_counts().plot.pie(figsize=(7, 7), 
                                       title="Event Pie Chart", 
                                       autopct='%1.1f%%', 
                                       legend=True)


# Now the matrices look pretty similar and we are ready to build the Recommendation Engine.

# In[101]:


recommendation_df.to_csv('./data/silver/userdata.csv', index=False)


# In[102]:


user_actions.to_csv('./data/silver/useractions.csv', index=False)


# If we look closely how event outcome is related to gender or income we can notice that correlation is quite weak, so other additional parameters should be definitely be taken into account.
