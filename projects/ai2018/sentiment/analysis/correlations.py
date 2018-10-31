#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   correlations.py
#        \author   chenghuige  
#          \date   2018-10-25 11:16:14.268580
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import numpy as np
import pandas as pd 
from scipy.stats import ks_2samp

import glob

import melt
import gezi

from tqdm import tqdm

import matplotlib.pyplot as plt 
import itertools


ATTRIBUTES = ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find',
              'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed', 
              'price_level', 'price_cost_effective', 'price_discount', 
              'environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness',
              'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
              'others_overall_experience', 'others_willing_to_consume_again']

#http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


if len(sys.argv) > 2:
  file1 = sys.argv[1]
  file2 = sys.argv[2]

  df1 = pd.read_csv(file1)
  df1 = df1.sort_values('id')
  df2 = pd.read_csv(file2)
  df2 = df2.sort_values('id')

  scores1 = [gezi.str2scores(x) for x in df1['score'].values]
  scores2 = [gezi.str2scores(x) for x in df2['score'].values]

  scores1 = np.reshape(scores1, [-1, len(ATTRIBUTES), 4])
  scores1 = gezi.softmax(scores1)
  scores2 = np.reshape(scores2, [-1, len(ATTRIBUTES), 4])
  scores2 = gezi.softmax(scores2)

  ndf1 = pd.DataFrame()
  ndf2 = pd.DataFrame()

  for i, attr in enumerate(ATTRIBUTES):
    score1 = np.reshape(scores1[:, i, :], [-1])
    score2 = np.reshape(scores2[:, i, :], [-1])
    ndf1[attr] = score1 
    ndf2[attr] = score2

    print('Attr:----------------------------------------------------%s' % attr)
    print(' Pearson\'s correlation score: %0.6f' %
          ndf1[attr].corr(
              ndf2[attr], method='pearson'))
    # print(' Kendall\'s correlation score: %0.6f' %
    #       ndf1[attr].corr(
    #           ndf2[attr], method='kendall'))
    # print(' Spearman\'s correlation score: %0.6f' %
    #       ndf1[attr].corr(
    #           ndf2[attr], method='spearman'))
    # ks_stat, p_value = ks_2samp(ndf1[attr].values,
    #                             ndf2[attr].values)
    # print(' Kolmogorov-Smirnov test:    KS-stat = %.6f    p-value = %.3e'
    #       % (ks_stat, p_value))
  print('----------------------------------------------------------all')
  scores1 = np.reshape(scores1, [-1])
  scores2 = np.reshape(scores2, [-1])
  ndf1, ndf2 = pd.DataFrame(), pd.DataFrame()
  ndf1['mean'] = scores1 
  ndf2['mean'] = scores2
  print(ndf1['mean'].corr(ndf2['mean'], method='pearson'))
else:
  #input is dir
  dir = sys.argv[1] if len(sys.argv) > 1 else './'
  if not os.path.isdir(dir):
    dir = './'
    infile = sys.argv[1]
    object_name = os.path.basename(infile).replace('.valid.csv', '').split('_ckpt-')[0].split('_model.ckpt-')[0]
  else:
    object_name = None

  object = None
  models = []
  dfs = []
  for file in glob.glob('%s/*.valid.csv' % dir):
    df = pd.read_csv(file)
    df = df.sort_values('id')
    scores = [gezi.str2scores(x) for x in df['score'].values]
    scores = np.reshape(scores, [-1, len(ATTRIBUTES), 4])
    scores = gezi.softmax(scores)
    ndf = pd.DataFrame()
    ndf['score'] = np.reshape(scores, [-1])
    dfs.append(ndf)
    model_name = os.path.basename(file).replace('.valid.csv', '').split('_ckpt-')[0].split('_model.ckpt-')[0]
    if model_name == object_name:
      object = model_name
    models.append(model_name)

  def calc_correlation(x, y, method):
    if method.startswith('ks'):
      ks_stat, p_value = ks_2samp(x, y)
      if method == 'ks_s':
        score = ks_stat
      else:
        score = p_value  
    else:
      score = x.corr(y, method=method)
    return score

  len_ = len(dfs)
  cm = np.zeros([len_, len_])
  methods = ['pearson', 'kendall', 'spearman', 'ks_s', 'ks_p']
  methods = methods[:1]
  for method in methods:
    print('---------------------------------------', method)
    for i in tqdm(range(len_), ascii=True):
      if object and models[i] != object:
        continue
      print('--------------------', models[i])
      for j in range(len_):
        cm[i, j] = calc_correlation(dfs[i]['score'], dfs[j]['score'], method=method)
      indexes = np.argsort(-cm[i])  
      for index in indexes:
        print(models[index], cm[i, index])

#   plt.figure()
#   plot_confusion_matrix(cm, classes=models)
#   plt.show()
    
