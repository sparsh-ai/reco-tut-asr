#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
project_name = "reco-tut-asr"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)

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


# In[8]:


import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


# In[25]:


items = pd.read_csv('./data/silver/items.csv')
items.head()


# In[26]:


actual_ratings = pd.read_csv('./data/silver/ratings.csv')
actual_ratings.head()


# In[27]:


cbf = pd.read_csv('./data/gold/cbf.csv')
item_item = pd.read_csv('./data/gold/item-item.csv')
user_user = pd.read_csv('./data/gold/user-user.csv')
pers_bias = pd.read_csv('./data/gold/pers-bias.csv')
mf = pd.read_csv('./data/gold/mf.csv')


# In[28]:


# preprocess
cbf = cbf.apply(lambda col: col.apply(lambda elem: str(elem).replace(',', '.'))).astype(float)
user_user = user_user.apply(lambda col: col.apply(lambda elem: str(elem).replace(',', '.'))).astype(float)
item_item = item_item.apply(lambda col: col.apply(lambda elem: str(elem).replace(',', '.'))).astype(float)
mf = mf.apply(lambda col: col.apply(lambda elem: str(elem).replace(',', '.'))).astype(float)
pers_bias = pers_bias.apply(lambda col: col.apply(lambda elem: str(elem).replace(',', '.'))).astype(float)


# In[29]:


recs = [cbf, item_item, user_user, pers_bias, mf]
recs_names = ['cbf', 'item_item', 'user_user', 'pers_bias', 'mf']


# ## Metrics

# In[30]:


def get_ratings(user_id):
    user_ratings = ratings[user_id]
    actual_ratings = user_ratings[~np.isnan(user_ratings)]
    return actual_ratings


def get_top_n(user_id, n):
    top_n = {}
    for rec, rec_name in zip(recs, recs_names):
        top_n_items = rec[user_id].argsort().sort_values()[:n].index.values
        top_n[rec_name] = top_n_items
    return top_n


def get_popular_items(n):
    pop_percentages = ratings.copy()
    pop_percentages['popularity'] = ratings.apply(lambda row: np.sum(~np.isnan(row))-1, axis=1)/len(ratings.columns[1::])
    pop_percentages = pop_percentages.sort_values(by = 'popularity', ascending=False)
    return pop_percentages.item.values[:n]


def get_rmse(user_id):   
    user_ratings = get_ratings(user_id)
    rmse = {}
    for rec, rec_name in zip(recs, recs_names):
        predicted_ratings = rec.loc[user_ratings.index, user_id]
        temp = np.sqrt(np.average((predicted_ratings - user_ratings)**2))
        rmse[rec_name] = temp
    return rmse


def get_precision_at_n(user_id, n):
    top_n = get_top_n(user_id, n)
    user_ratings = get_ratings(user_id).index.values
    precisions = {}
    for rec, rec_name in zip(recs, recs_names):
        temp = np.sum(np.isin(top_n[rec_name], user_ratings))/n
        precisions[rec_name] = temp
    return precisions


# We will use the "FullCat" column in the items catalog to determine the product diversity in the recommendations.
# The recommender with a high number of distinct product categories in its recommendations is said to be product-diverse
def get_product_diversity(user_id, n):
    top_n = get_top_n(user_id, n)
    product_diversity = {}
    for rec_name in top_n:
        categories = items.loc[top_n[rec_name]][['FullCat']].values
        categories = set([item for sublist in categories for item in sublist])
        product_diversity[rec_name] = len(categories)
    return product_diversity


# We will use the "Price" column in the items catalog to determine cost diversity in the recommendations.
# The recommender with a high standard deviation in the cost across all its recommendations is said to be cost-diverse
def get_cost_diversity(user_id, n):
    top_n = get_top_n(user_id,n)
    cost_diversity = {}
    for rec_name in top_n:
        std_dev = np.std(items.loc[top_n[rec_name]][['Price']].values)
        cost_diversity[rec_name] = std_dev
    return cost_diversity


# We will use inverse popularity as a measure of serendipity.
# The recommender with least number of recommendations on the "most popular" list, will be called most serendipitous
def get_serendipity(user_id, n):
    top_n = get_top_n(user_id,n)
    popular_items = get_popular_items(20)
    serendipity = {}
    for rec, rec_name in zip(recs, recs_names):
        popularity = np.sum(np.isin(top_n[rec_name],popular_items))
        if int(popularity) == 0:
            serendipity[rec_name] = 1
        else:
            serendipity[rec_name] = 1/popularity
    return serendipity


# In[31]:


avg_metrics = {}
for name in recs_names: 
    avg_metrics[name] = {"rmse": [], "precision_at_n": [], "product_diversity": [], "cost_diversity": [], "serendipity": []}

for user_id in ratings.columns:
    if user_id == 'item':
        continue
    user_id = str(user_id)
    rmse = get_rmse(user_id)
    precision_at_n = get_precision_at_n(user_id, 10)
    product_diversity = get_product_diversity(user_id, 10)
    cost_diversity = get_cost_diversity(user_id, 10)
    serendipity = get_serendipity(user_id, 10)
    for key in avg_metrics:
        rec_name = avg_metrics[key]
        rec_name['rmse'].append(rmse[key])
        rec_name['precision_at_n'].append(precision_at_n[key])
        rec_name['product_diversity'].append(product_diversity[key])
        rec_name['cost_diversity'].append(cost_diversity[key])
        rec_name['serendipity'].append(serendipity[key])

# The Price for certain items is not available. Also rmse for certain users is turning out to be NaN.
# Ignoring nans in the average metric calculation for now. So basically narrowing down the evaluation to users who have
# rated atleast one item and items for which the price is known.
for key in avg_metrics:
    rec_name = avg_metrics[key]
    for metric in rec_name:
        temp = rec_name[metric]
        temp = [x for x in temp if not np.isnan(x)]
        rec_name[metric] = sum(temp) / len(temp)


# In[32]:


avg_metrics


# ## Hybridization

# In[33]:


# Creating a dataframe with ratings from all algorithms and user_ratings as ground truth
users = []
items = []
user_ratings = []
cbf_ratings = []
user_user_ratings = []
item_item_ratings = []
mf_ratings = []
pers_bias_ratings = []

for user_id in ratings.columns:
    if user_id == 'item':
        continue
    user_id = str(user_id)
    true_ratings = get_ratings(user_id)
    user_ratings.extend(true_ratings.values)
    users.extend([user_id]*len(true_ratings))
    items.extend(ratings.loc[true_ratings.index].item.values)
    cbf_ratings.extend(cbf.loc[true_ratings.index, user_id].values)
    item_item_ratings.extend(item_item.loc[true_ratings.index, user_id].values)
    user_user_ratings.extend(user_user.loc[true_ratings.index, user_id].values)
    pers_bias_ratings.extend(pers_bias.loc[true_ratings.index, user_id].values)
    mf_ratings.extend(mf.loc[true_ratings.index, user_id].values)
    
df = pd.DataFrame({'user': users, 'item': items,'true_rating': user_ratings, 'cbf':cbf_ratings, 'item_item':item_item_ratings, 'user_user': user_user_ratings, 'pers_bias':pers_bias_ratings, 'mf':mf_ratings})


# In[34]:


df = df.dropna()


# In[35]:


df.head()


# ### Linear Combination

# In[39]:


clf = LinearRegression()


# In[40]:


# Split data in 80-20 train and test sets
train = df[0:(int(0.8*len(df)))]
test = df[(int(0.8*len(df)))::]


# In[41]:


train_data = train.drop(['user', 'item','true_rating'], axis=1)
train_labels = train.true_rating.values
model = clf.fit(train_data, train_labels)


# In[42]:


test_data = test.drop(['user', 'item','true_rating'], axis=1)
test_labels = test.true_rating.values
predictions = model.predict(test_data)


# In[44]:


# Avg RMSE predictions
avg_rmse = np.sqrt(np.average((predictions - test_labels)**2))
avg_rmse


# #### Top 5 for three users

# In[46]:


# Pick three users
users = random.sample(list(ratings.columns[1::]), 3)
print(users)


# In[47]:


train_data = df.drop(['user', 'item','true_rating'], axis=1)
train_labels = df.true_rating.values
model = clf.fit(train_data, train_labels)


# In[ ]:


top_5 = {}
for user in users:
    df_preds = df[df.user == user]
    preds = model.predict(df_preds.drop(['user', 'item','true_rating'], axis=1))
    df_preds['predictions'] = preds
    top_5_items = list(df_preds.sort_values(by=['predictions'], ascending=False)[:5].item.values)
    top_5[user] = top_5_items


# In[49]:


top_5


# ### Non-linear Combination 

# For a non-linear combination of the algorithms, we'll use the DecisionTreeRegressor method in scikitlearn

# In[51]:


clf = DecisionTreeRegressor()


# In[52]:


# Split data in 80-20 train and test sets
train = df[0:(int(0.8*len(df)))]
test = df[(int(0.8*len(df)))::]


# In[53]:


train_data = train.drop(['user', 'item','true_rating'], axis=1)
train_labels = train.true_rating.values
model = clf.fit(train_data, train_labels)


# In[54]:


test_data = test.drop(['user', 'item','true_rating'], axis=1)
test_labels = test.true_rating.values
predictions = model.predict(test_data)


# In[55]:


# Avg RMSE predictions
avg_rmse = np.sqrt(np.average((predictions - test_labels)**2))
avg_rmse


# #### Top-5 for 3 users

# In[56]:


# Using the same users as above to compare across the same users
users = ['3430', '112', '1817']


# In[57]:


train_data = df.drop(['user', 'item','true_rating'], axis=1)
train_labels = df.true_rating.values
model = clf.fit(train_data, train_labels)


# In[ ]:


top_5 = {}
for user in users:
    df_preds = df[df.user == user]
    preds = model.predict(df_preds.drop(['user', 'item','true_rating'], axis=1))
    df_preds['predictions'] = preds
    top_5_items = list(df_preds.sort_values(by=['predictions'], ascending=False)[:5].item.values)
    top_5[user] = top_5_items


# In[59]:


top_5


# ## Different recommenders based on user type

# This hybridization techniques aims to create separate recomemnder strategies for two separate scenarios- one where users end up on the Nile-River.com landing page via banner ads for school products and other where users arrive at the landing page via endoresements for office products. For the first scenario, we'll pick a 3:2 ratio of school (inexpensive) products vs. office (expensive) products and the reverse for the second scenario i.e. 2:3 ratio of school to office products. Here we will show the evaluate only for the first scenario.

# In[68]:


# Determine threshold to label an item cheap or expensive- let's set this as the third quantile of the price list
# This is assuming office products are mostly in the expensive bracket
items = pd.read_csv('./data/silver/items.csv') # df converted to list in processing above, so loading back
prices = items.Price.values
price_threshold = np.percentile([x for x in prices if not np.isnan(x)], 75)


# ### Performance

# In[69]:


def get_precision_at_n(user_id, top_n):
    user_ratings = get_ratings(user_id).index.values
    precision_at_n = np.sum(np.isin(top_n, user_ratings))/ len(top_n)
    return precision_at_n


# In[70]:


def get_cost_diversity(top_n):
    std_dev = np.std(items.loc[top_n][['Price']].values)
    return std_dev


# In[71]:


def get_product_diversity(top_n):
    categories = items.loc[top_n][['FullCat']].values
    categories = set([item for sublist in categories for item in sublist])
    return len(categories)


# In[72]:


def get_serendipity(top_n):
    popular_items = get_popular_items(20)
    popularity = np.sum(np.isin(top_n,popular_items))
    if int(popularity) == 0:
        serendipity = 1
    else:
        serendipity = 1/popularity
    return serendipity


# In[73]:


# To pick which items to finally recommend, let's assume that all the items in the top-5 for each recommender are 
# equally relevant. We can potentially include some ranking based selection to pick item that are more relavant AND fit the
# cost criteria. For now, we'll pick at random since we're assuming all items are equally relevant.
def get_mixed_recs(user_id, n, n_cheap, n_exp):
    top_n_overall_items = []
    top_n_overall_prices = []
    mixed_recs = [] 
    for rec, rec_name in zip(recs, recs_names):
        top_n_items = rec[user_id].argsort().sort_values()[:n].index.values
        top_n_prices = items.loc[top_n_items][['Price']].values
        top_n_overall_items.extend(top_n_items)
        top_n_overall_prices.extend(top_n_prices)
    top_dict = dict(zip(top_n_overall_items, top_n_overall_prices))
    top_cheap = dict(filter(lambda elem: elem[1] <= price_threshold, top_dict.items())).keys()
    top_exp = dict(filter(lambda elem: elem[1] > price_threshold, top_dict.items())).keys()
    mixed_recs = random.sample(list(top_cheap), n_cheap) + random.sample(list(top_exp), n_exp)
    return mixed_recs


# In[74]:


avg_metrics = {"precision_at_n": [], "product_diversity": [], "cost_diversity": [], "serendipity": []}
for user_id in ratings.columns:
    if user_id == 'item':
        continue
    user_id = str(user_id)
    top_5 = get_mixed_recs(user_id, 5, 3, 2)
    avg_metrics["precision_at_n"].append(get_precision_at_n(user_id, top_5))
    avg_metrics["cost_diversity"].append(get_cost_diversity(top_5))
    avg_metrics["product_diversity"].append(get_product_diversity(top_5))
    avg_metrics["serendipity"].append(get_serendipity(top_5))

for metric in avg_metrics:
    temp = avg_metrics[metric]
    temp = [x for x in temp if not np.isnan(x)]
    avg_metrics[metric] = sum(temp) / len(temp)


# In[75]:


avg_metrics


# ### Top-5 for three users

# In[76]:


# Assuming all three users ended up on the landing pagee through scenario 1 i.e. banner ads for school products
users = ['3430', '112', '1817']


# In[77]:


top_5 = {}
for user_id in users:
    # For office products
    # top_5[user_id] = get_mixed_recs(user_id, 5, 2, 3)
    # For school products
    top_5[user_id] = list(ratings.loc[get_mixed_recs(user_id, 5, 3, 2)].item.values)


# In[78]:


top_5


# ## Switching hybridization

# We will not be implementing this hybridizaton as such, but we will explore whether or not the strategy of using content based filtering for new users (users with fewer/no ratings) or items with less ratings is even reasonable for this dataset. For this, let's begin with visualizing the number of ratings for the users in the dataset.

# In[80]:


item_ratings = ratings.apply(lambda row: np.sum(~np.isnan(row))-1, axis=1)


# In[81]:


plt.hist(item_ratings)
plt.xlabel("Number of ratings")
plt.ylabel("number of items")


# In[82]:


# Number of items with < 10 ratings
count_less_than_10 = np.count_nonzero(item_ratings<10)/len(item_ratings)*100


# In[83]:


count_less_than_10


# In[84]:


user_ratings = []
for user_id in ratings.columns:
    if user_id == 'item':
        continue
    user_id = str(user_id)
    user_ratings.append(len(get_ratings(user_id)))


# In[85]:


plt.hist(user_ratings)
plt.xlabel("Number of ratings")
plt.ylabel("number of users")


# In[86]:


# Number of users with < 10 ratings
count_less_than_10 = np.count_nonzero(np.array(user_ratings)<10)/len(user_ratings)*100


# In[87]:


count_less_than_10

