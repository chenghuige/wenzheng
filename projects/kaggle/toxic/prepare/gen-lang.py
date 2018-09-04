#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-lang.py
#        \author   chenghuige  
#          \date   2018-03-11 22:49:32.208144
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import pandas as pd

import tqdm 

import numpy as np


from langdetect import detect_langs
from langdetect import DetectorFactory
DetectorFactory.seed = 0

def run(file_):
  ofile = file_.replace('.csv', '.langs.csv')
  #print(ofile)
  df = pd.read_csv(file_) 
  #df = df[:100]
  ids = df['id'].values
  comments = df['comment_text'].values 

  results = []
  for comment in tqdm.tqdm(comments):
    try:
      result = detect_langs(comment.lower())
      l = ['%s:%f' %(x.lang, x.prob) for x in result]
      results.append(' '.join(l))
    except Exception:
      results.append('None')
    
  results = np.array(results)

  odf = pd.DataFrame(data=ids, columns=['id'])
  odf['lang'] = results
  odf.to_csv(ofile, index=False)

run('~/data/kaggle/toxic/train.csv')
run('~/data/kaggle/toxic/test.csv')
