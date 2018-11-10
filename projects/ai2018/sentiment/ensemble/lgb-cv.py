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

from tqdm import tqdm

from sklearn.metrics import f1_score 

ATTRIBUTES = ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find',
              'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed', 
              'price_level', 'price_cost_effective', 'price_discount', 
              'environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness',
              'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
              'others_overall_experience', 'others_willing_to_consume_again']
num_attrs = len(ATTRIBUTES)
num_classes = 4

idx = 2

results = None

df2 = pd.DataFrame()
idf2 = pd.DataFrame()

valid_file = './ensemble.valid.csv'
infer_file = './ensemble.infer.debug.csv'

idf = pd.read_csv(infer_file)
idf = idf.sort_values('id')
iscores = idf['logit']
iscores = [gezi.str2scores(score) for score in iscores]
# iscores2 = []
# for score in iscores: 
#   score = gezi.softmax(np.reshape(score, [num_attrs, 4]), -1)
#   score = np.reshape(score, [-1])
#   iscores2.append(score)
# iscores = iscores2
iscores = np.array(iscores)

print(valid_file)
df = pd.read_csv(valid_file, sep=',')
df = df.sort_values('id')
labels = df.iloc[:,idx:idx+num_attrs].values
predicts = df.iloc[:,idx+num_attrs:idx+2*num_attrs].values
#scores = df['score']
scores = df['logit']
#scores = df['prob']
scores = [gezi.str2scores(score) for score in scores]
# scores2 = []
# for score in scores: 
#   score = gezi.softmax(np.reshape(score, [num_attrs, 4]), -1)
#   score = np.reshape(score, [-1])
#   scores2.append(score)
# scores = scores2
scores = np.array(scores)
ids = df.iloc[:,0].values 

cnames = []
for attr in ATTRIBUTES:
  for i in range(4):
    cnames.append(f'{attr}_{i}')

print('---------', cnames)  
df2['id'] = ids
idf2['id'] = ids
for i, label in enumerate(df.columns.values[idx:idx+num_attrs]):
  df2[label] = labels[:,i]

for i, name in enumerate(cnames):
  df2[name] = scores[:,i]
  idf2[name] = iscores[:, i]

print("Light Gradient Boosting Classifier: ")

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

clf = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.05, objective='multiclass',
                         random_state=314, silent=True, metric='None', 
                         n_jobs=4, n_estimators=5000, class_weight='balanced')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.metrics import f1_score
def evaluate_macroF1_lgb(truth, predictions):  
    # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483
    pred_labels = predictions.reshape(len(np.unique(truth)),-1).argmax(axis=0)
    f1 = f1_score(truth, pred_labels, average='macro')
    return ('macroF1', f1, True) 

import lightgbm as lgb

def learning_rate_power_0997(current_iter):
    base_learning_rate = 0.1
    min_learning_rate = 0.02
    lr = base_learning_rate  * np.power(.995, current_iter)
    return max(lr, min_learning_rate)

# attr = 'others_willing_to_consume_again'
# attr = 'others_overall_experience'
# i = ATTRIBUTES.index(attr)

for i, attr in tqdm(enumerate(ATTRIBUTES), ascii=True):
  index = 1 + num_attrs + i * 4
  index2 = 1 + i * 4
  X_ = df2.iloc[:, index: index + 4]
  iX_ = idf2.iloc[:, index2: index2 + 4]

  f1_list = []
  K = 8
  y = df2[attr + '_y'] + 2

  ix_ = iX_.values
  X = X_.values
  y = y.values
  kf = KFold(n_splits=K)
  kf.get_n_splits(X)
  preds = []
  ipreds = []
  ys = []
  for fold, (train_index, valid_index) in enumerate(kf.split(X)):
    print(f'{i}:{attr} FOLD:{fold}')
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    ys.append(y_valid)

    #print(X_train.shape, y_train.shape)
    
    fit_params={"early_stopping_rounds":300, 
                "eval_metric" : evaluate_macroF1_lgb, 
                "eval_set" : [(X_valid,y_valid)],
                'eval_names': ['valid'],
                #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
                'verbose': False,
                'categorical_feature': 'auto'}

    fit_params['callbacks'] = [lgb.reset_parameter(learning_rate=learning_rate_power_0997)]
    
    opt_parameters = {
                      #'colsample_bytree': 0.9221304051471293, 
                      'min_child_samples': 150, 
                      'num_leaves': 2, 
                      #'subsample': 0.9510118790770111, 
                      'class_weight': 'balanced', 
                      'lambda_l1': 1.79,
                      'lambda_l2': 1.71,
                      'num_trees': 2000
                      }
    #clf_final = lgb.LGBMClassifier(**clf.get_params())
    #clf_final.set_params(**opt_parameters)
    clf_final = lgb.LGBMClassifier(bagging_fraction=0.9957236684465528, boosting_type='gbdt',
        class_weight='balanced', colsample_bytree=0.7953949538181928,
        feature_fraction=0.7333800304661316, lambda_l1=1.79753950286893,
        lambda_l2=1.710590311253639, learning_rate=0.2, max_depth=6,
        metric='None', min_child_samples=48,
        min_child_weight=48.94067592560281,
        min_split_gain=0.016737988780906453, n_estimators=5000, n_jobs=4,
        num_leaves=34, objective='multiclass', random_state=None,
        reg_alpha=0.0, reg_lambda=0.0, silent=True,
        subsample=0.9033449610388691, subsample_for_bin=200000,
        subsample_freq=1)
    clf_final.set_params(**opt_parameters)

    def learning_rate_power_0997(current_iter):
        base_learning_rate = 0.1
        min_learning_rate = 0.02
        lr = base_learning_rate  * np.power(.997, current_iter)
        return max(lr, min_learning_rate)

    #Train the final model with learning rate decay
    fit_params['verbose'] = 200
    _ = clf_final.fit(X_train, y_train, **fit_params)

    preds.append(clf_final.predict(X_valid))

  preds = np.concatenate(preds)
  y = np.concatenate(ys)

  df[attr] = np.array(preds) - 2
  ipreds = clf_final.predict(iX_)
  idf[attr] = np.array(ipreds) - 2

df.to_csv('ensemble.valid.lgb.csv', index=False)
idf.to_csv('ensemble.infer.lgb.csv', index=False)
# attr = 'others_overall_experience'
# #f1 = f1_score(labels[:,i] + 2, preds, average='macro')
# f1 = f1_score(y, preds, average='macro')
# f1_list.append(f1)
# print(attr, f1)

# f1 = np.mean(f1_list)

# for i, attr in enumerate(ATTRIBUTES):
#   print(attr, f1_list[i])
# print('f1:', f1)


