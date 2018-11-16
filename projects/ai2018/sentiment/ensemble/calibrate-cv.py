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

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

from sklearn.model_selection import KFold

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

# idf = pd.read_csv(infer_file)
# idf = idf.sort_values('id')
#iscores = idf['logit']
#iscores = [gezi.str2scores(score) for score in iscores]
# iscores2 = []
# for score in iscores: 
#   score = gezi.softmax(np.reshape(score, [num_attrs, 4]), -1)
#   score = np.reshape(score, [-1])
#   iscores2.append(score)
# iscores = iscores2
#iscores = np.array(iscores)

print(valid_file)
df = pd.read_csv(valid_file, sep=',')
df = df.sort_values('id')
labels = df.iloc[:,idx:idx+num_attrs].values
predicts = df.iloc[:,idx+num_attrs:idx+2*num_attrs].values
scores = df['score']
#scores = df['logit']
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
#idf2['id'] = ids
for i, label in enumerate(df.columns.values[idx:idx+num_attrs]):
  df2[label] = labels[:,i]

for i, name in enumerate(cnames):
  df2[name] = scores[:,i]
  #idf2[name] = iscores[:, i]

for i, attr in tqdm(enumerate(ATTRIBUTES), ascii=True):
  index = 1 + num_attrs + i * 4
  index2 = 1 + i * 4
  X_ = df2.iloc[:, index: index + 4]
  #iX_ = idf2.iloc[:, index2: index2 + 4]

  f1_list = []
  K = 5
  y = df2[attr + '_y'] + 2

  #ix_ = iX_.values
  X = X_.values
  y = y.values
  kf = KFold(n_splits=K)
  kf.get_n_splits(X)
  preds = []
  ipreds = []
  ys = []
  for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f'{i}:{attr} FOLD:{fold}')
    X_train, X_valid, X_test = X[train_index[:-3000]], X[train_index[-3000:]], X[test_index]
    y_train, y_valid, y_test = y[train_index[:-3000]], y[train_index[-3000:]], y[test_index]
    ys.append(y_test)

    # Train uncalibrated random forest classifier on whole train and validation
    # data and evaluate on test data
    clf = RandomForestClassifier(n_estimators=25)
    clf.fit(np.concatenate([X_train, X_valid], 0), np.concatenate([y_train, y_valid], 0))
    clf_probs = clf.predict_proba(X_test)
    score = log_loss(y_test, clf_probs)

    print('1', score)

    # Train random forest classifier, calibrate on validation data and evaluate
    # on test data
    clf = RandomForestClassifier(n_estimators=25)
    clf.fit(X_train, y_train)
    clf_probs = clf.predict_proba(X_test)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
    sig_clf.fit(X_valid, y_valid)
    sig_clf_probs = sig_clf.predict_proba(X_test)
    sig_score = log_loss(y_test, sig_clf_probs)

    print('2', sig_score)

    preds.append(clf.predict(X_test))

  preds = np.concatenate(preds)
  y = np.concatenate(ys)

  df[attr] = np.array(preds) - 2
  #ipreds = clf_final.predict(iX_)
  #idf[attr] = np.array(ipreds) - 2

df.to_csv('ensemble.valid.calibrate.csv', index=False)
#idf.to_csv('ensemble.infer.calibrate.csv', index=False)

