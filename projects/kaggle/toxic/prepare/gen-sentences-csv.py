#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-sentences.py
#        \author   chenghuige  
#          \date   2018-03-19 13:28:51.297962
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import pandas as pd 
import gezi
import multiprocessing as mp 
import numpy as np

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
input = None

def run():
  sent_file = input.replace('.csv', '.sents.txt')
  print(sent_file)
  m = {}
  for line in open(sent_file):
    id, sent = line.rstrip('\n').split('\t', 1)
    sent = sent.replace('NEWLINE', '\n')
    if len(sent.strip()) < 3:
      continue
    if id not in m:
      m[id] = [sent]
    else:
      m[id].append(sent)
    
  df = pd.read_csv(input)
  ids = df['id'].values
  comments = df['comment_text'].values 
  if 'train' in input:
    labels = df[CLASSES].values
  else:
   labels = [[0.] * 6] * len(df)


  ids_ = []
  comments_ = []
  labels_ = []

  output = input.replace('.csv', '.sents.csv')
  print(output)
  num = 0
  for id, comment, label in zip(ids, comments, labels):
    if num % 1000 == 0:
      print(num)
    num += 1
    if id not in m:
      sents = ['ok']
    else:
      sents = m[id]
    for sent in sents:
      ids_.append(id)
      #print('sent', sent)
      comments_.append(sent)
      labels_.append(label)

  print(len(labels_), len(comments_))
  ids = np.array(ids_)
  print('ids ok')
  labels = np.array(labels_)
  print('labels ok')
  #comments = np.array(comments_)
  comments = comments_
  print('comments ok')
  #print(comments)

  odf = pd.DataFrame(data=labels, columns=CLASSES)
  odf['comment_text'] = comments
  odf['id'] = ids
  odf = odf[['id', 'comment_text'] + CLASSES]
  odf.to_csv(output, index=False)

#input = '/home/gezi/data/kaggle/toxic/train.csv'
#run()
input = '/home/gezi/data/kaggle/toxic/test.csv'
run()

