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

def run(file_):
  ofile = file_.replace('langs.csv', 'enprob.csv')
  #print(ofile)
  df = pd.read_csv(file_) 
  #df = df[:100]
  ids = df['id'].values
  langs = df['lang'].values 
  
  en_probs = []

  for item in langs:
    if item == 'None':
      en_probs.append(1.)
    else:
      l = item.split()
      if len(l) == 1 and l[0].startswith('en:') and float(l[0].split(':')[1]) > 0.999:
        en_probs.append(1.)
      else:
        en_prob = 0.
        for x in l:
          lang, prob = x.split(':')
          prob = float(prob)
          if lang == 'en':
            en_prob = prob 
        en_probs.append(en_prob)
    
  en_probs = np.array(en_probs)

  odf = pd.DataFrame(data=ids, columns=['id'])
  odf['enprob'] = en_probs
  odf.to_csv(ofile, index=False)

run('~/data/kaggle/toxic/train.langs.csv')
run('~/data/kaggle/toxic/test.langs.csv')
