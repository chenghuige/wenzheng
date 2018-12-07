#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   evaluate.py
#        \author   chenghuige  
#          \date   2018-12-07 15:19:16.505094
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import gezi

classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def evaluate(labels, logits, ids=None):
  logits = logits[:, :len(classes)]
  predicts = np.argmax(logits, -1)
  acc = np.mean(np.equal(predicts, labels))

  probs = gezi.softmax(logits)
  loss = log_loss(labels, probs)

  vals = [loss, acc]
  names = ['loss', 'acc']

  return vals, names

def write(ids, labels, logits, ofile):
  logits = logits[:, :len(classes)]
  df = pd.DataFrame()
  df['id'] = ids
  predicts = np.argmax(logits, -1)
  df['predict'] = [classes[x] for x in predicts]
  if labels is not None: 
    df['label'] = [classes[x] for x in labels]
  df= df.sort_values('id') 
  df.to_csv(ofile, index=False)

def valid_write(ids, labels, logits, ofile):
  return write(ids, labels, logits, ofile)

def infer_write(ids, logits, ofile):
  return write(ids, None, logits, ofile)
