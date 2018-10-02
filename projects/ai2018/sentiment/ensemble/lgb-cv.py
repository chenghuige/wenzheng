#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   ensemble.py
#        \author   chenghuige  
#          \date   2018-09-15 19:04:21.026718
#   \Description  now work well, worse then mean...
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import glob
import pandas as pd
import numpy as np 
from sklearn.metrics import f1_score 
#from sklearn.preprocessing import minmax_scale
import gezi

import matplotlib.pyplot as plt

import lightgbm as lgb

from sklearn.metrics import f1_score 

idir = sys.argv[1]
ATTRIBUTES = ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find',
              'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed', 
              'price_level', 'price_cost_effective', 'price_discount', 
              'environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness',
              'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
              'others_overall_experience', 'others_willing_to_consume_again']
num_attrs = len(ATTRIBUTES)
num_classes = 4

def parse(l):
  return np.array([float(x.strip()) for x in l[1:-1].split(',')])

idx = 2

results = None

df2 = pd.DataFrame()
cnames = []

for fid, file_ in enumerate(glob.glob('%s/*.valid.csv' % idir)):
  print(file_)
  df = pd.read_csv(file_, sep=',')
  df.sort_index(0, inplace=True)
  labels = df.iloc[:,idx:idx+num_attrs].values
  predicts = df.iloc[:,idx+num_attrs:idx+2*num_attrs].values
  scores = df['score']
  scores = [parse(score) for score in scores]
  scores2 = []
  for score in scores: 
    score = gezi.softmax(np.reshape(score, [num_attrs, 4]), -1)
    score = np.reshape(score, [-1])
    scores2.append(score)
  scores = scores2
  scores = np.array(scores)
  ids = df.iloc[:,0].values 

  cname = '.'.join(os.path.basename(file_).split('.')[:-2])
  #cnames = [cname + '_' + str(x) for x in range(num_attrs)]
  cnames = []
  for attr in ATTRIBUTES:
    for i in range(4):
      cnames.append(f'{attr}_{i}_{cname}')
  
  if fid == 0:
    df2['id'] = ids
    for i, label in enumerate(df.columns.values[idx:idx+num_attrs]):
      df2[label] = labels[:,i]


  #df_result[cnames] = scores
  for i, name in enumerate(cnames):
    #if name.startswith('others_willing_to_consume_again_'):
    if name.startswith('others_overall_experience'):
      df2[name] = scores[:,i]

#df2.to_csv(f'{idir}/result.csv', index=False)


print("Light Gradient Boosting Classifier: ")
# params =  {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'multiclass',
#     'num_class': 4,
#     'metric': ['multi_logloss'],
#     "learning_rate": 0.05,
#      "num_leaves": 8,
#      "max_depth": 4,
#      "num_trees": 20,
#      "feature_fraction": 0.9,
#      "bagging_fraction": 0.8,
#      "reg_alpha": 0.15,
#      "reg_lambda": 0.15,
# #      "min_split_gain": 0,
#       "min_child_weight": 5
#                 }
params =  {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': num_classes,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'verbose': -1,
        'metric': ['multi_logloss'],
        "learning_rate": 0.2,
        "max_depth": 5,
        "num_leaves": 10,
        "reg_lambda": 0.1,
        "num_trees": 500,
        "min_data_in_leaf": 100,
          }


# lgb_cv = lgb.cv(
#     params = lgbm_params,
#     train_set = lgtrain,
#     num_boost_round=10,
#     stratified=True,
#     nfold = 5,
#     verbose_eval=1,
#     seed = 23,
#     early_stopping_rounds=75)

# loss = lgbm_params["metric"][0]
# optimal_rounds = np.argmin(lgb_cv[str(loss) + '-mean'])
# best_cv_score = min(lgb_cv[str(loss) + '-mean'])

# print("\nOptimal Round: {}\nOptimal Score: {} + {}".format(
#     optimal_rounds,best_cv_score,lgb_cv[str(loss) + '-stdv'][optimal_rounds]))

# results = pd.DataFrame(columns = ["Rounds","Score","STDV", "LB", "Parameters"])
# results = results.append({"Rounds": optimal_rounds,
#                           "Score": best_cv_score,
#                           "STDV": lgb_cv[str(loss) + '-stdv'][optimal_rounds],
#                           "LB": None,
#                           "Parameters": lgbm_params}, ignore_index=True)

# with open(f'{idir}/results.csv', 'a') as f:
#     results.to_csv(f, header=False)


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

#X_train, X_valid, y_train, y_valid = train_valid_split(X, y, valid_size=0.33, random_state=42)


attr = 'others_willing_to_consume_again'
attr = 'others_overall_experience'
i = ATTRIBUTES.index(attr)

X_ = df2.iloc[:,idx+num_attrs:]
f1_list = []

#for i, attr in enumerate(ATTRIBUTES):
K = 5
y = df2[attr + '_y'] + 2
lgtrain = lgb.Dataset(X_, y, categorical_feature= "auto")

X = X_.values
y = y.values
kf = KFold(n_splits=K)
kf.get_n_splits(X)
preds = []
for fold, (train_index, valid_index) in enumerate(kf.split(X)):
  print('--------------------FOLD', fold, "TRAIN:", train_index, "valid:", valid_index)
  X_train, X_valid = X[train_index], X[valid_index]
  y_train, y_valid = y[train_index], y[valid_index]

  print(len(X_train), len(y_train))
  
  lgtrain = lgb.Dataset(X_train, y_train, categorical_feature= "auto")
  lgvalid = lgb.Dataset(X_valid, y_valid, categorical_feature= "auto")
  gbm = lgb.train(params,
                  lgtrain,
                  1000,
                  valid_sets=[lgtrain, lgvalid],
                  early_stopping_rounds=50,
                  verbose_eval=4)

  preds.append(gbm.predict(X_valid,
                          num_iteration=gbm.best_iteration))
  # Plot importance
  lgb.plot_importance(gbm)
  plt.show()

preds = np.concatenate(preds)

print(labels[:,0] + 2)
print(np.argmax(preds, 1))
f1 = f1_score(labels[:,i] + 2, np.argmax(preds, 1), average='macro')
f1_list.append(f1)
print(attr, f1)

f1 = np.mean(f1_list)

# for i, attr in enumerate(ATTRIBUTES):
#   print(attr, f1_list[i])
# print('f1:', f1)


