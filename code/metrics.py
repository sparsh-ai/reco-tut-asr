
import pandas as pd
import numpy as np


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