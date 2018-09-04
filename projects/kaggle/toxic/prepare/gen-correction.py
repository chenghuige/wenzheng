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

from textblob import Word

import gezi 
from gezi import Vocabulary

from tokenizer import is_toxic

from tokenizer import en_filter

def get_en_token(token):
  tokens, en_tokens = en_filter(token)
  en_token = ''.join(en_tokens)
  return en_token

vocab = Vocabulary('/home/gezi/temp/toxic/v13/tfrecords/glove/vocab.txt', max_words=3000)
#vocab = Vocabulary('/home/gezi/data/glove/glove-vocab.txt')

def run(file_):
  ofile = file_.replace('.csv', '.correction.csv')
  #print(ofile)
  df = pd.read_csv(file_) 
  df = df[:1000]
  ids = df['id'].values 
  try:
    toxic = df['toxic'].values
  except Exception:
    toxic = [0] * len(ids)
  comments = df['comment_text'].values 

  results = []
  i = 0
  for comment in tqdm.tqdm(comments):
    tokens = gezi.tokenize_filter_empty(comment)
    corrections = []
    for token in tokens:
      if vocab.has(token):
        continue

      token = get_en_token(token)
      if vocab.has(token):
        continue

      word = Word(token)
      l = word.spellcheck()
      w, prob = l[0]

      if w != token and token[0].lower() == w[0].lower() and prob >= 0.9:
        if is_toxic(w):
          print(ids[i], token, w, prob, toxic[i])
          corrections.append('%s:%s' % (token, w))
    
    result = ' '.join(corrections) if corrections else 'None'
    results.append(result)
    i += 1

  results = np.array(results)

  odf = pd.DataFrame(data=ids, columns=['id'])
  odf['correction'] = results
  odf['toxic'] = toxic
  odf.to_csv(ofile, index=False)

mode = 'train'
if len(sys.argv) > 1:
  mode = sys.argv[1]

if mode == 'train':
  run('~/data/kaggle/toxic/train.csv')
else:
  run('~/data/kaggle/toxic/test.csv')
