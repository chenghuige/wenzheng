#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   count-unks.py
#        \author   chenghuige  
#          \date   2018-03-21 00:04:00.722879
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

from gezi import Vocabulary
import pandas as pd

dir = '/home/gezi/temp/toxic/v16/tfrecords/glove.lower/'

vocab = Vocabulary(dir + 'vocab.txt')

def run(input):
  total_tokens = 0
  total_unks = 0
  num_specials = 0 
  num_toxic = 0
  output = input.replace('.csv', '.numunks.csv')
  output_speial = input.replace('.csv', '.special.csv')
  df = pd.read_csv(input)
  ids = df['id'].values
  comments = df['tokens'].values 
  if 'toxic' not in df.columns:
    df['toxic'] = [0.] * len(comments)
  toxics = df['toxic'].values
  
  num_tokens_list = []
  num_unks_list = []

  sids = []
  scoments = []
  sratios = []
  stoxics = []

  for id, comment, toxic in zip(ids, comments, toxics):
    tokens = comment.split()
    num_tokens = len(tokens)
    num_unks = len([x for x in tokens if not vocab.has(x.lower())])
    num_tokens_list.append(num_tokens)
    num_unks_list.append(num_unks)
    

    total_tokens += num_tokens
    total_unks += num_unks
    ratio = num_unks / num_tokens 
    is_special = False 

    if ratio > 0.5:
      is_special = True 
    if ratio > 0.25 and num_tokens > 10:
      is_special = True

    if is_special:
      num_specials += 1
      sids.append(id)
      scoments.append(comment)
      sratios.append(ratio)
      if toxic > 0:
        num_toxic += 1
      stoxics.append(toxic)

  odf = pd.DataFrame(data=ids, columns=['id'])
  odf['num_tokens'] = num_tokens_list
  odf['num_unks'] = num_unks_list 
  odf['toxic'] = df['toxic'].values

  odf.to_csv(output, index=False)

  odf = pd.DataFrame(data=sids, columns=['id'])
  odf['comment_text'] = scoments
  odf['unk_ratio'] = sratios 
  odf['toxic'] = stoxics
  odf.to_csv(output_speial, index=False)

  print('total_tokens', total_tokens, 'total_unks', total_unks, 'unk_ratio', total_unks / total_tokens, 'num_specials', num_specials, 'num_toxic', num_toxic)

run(dir + 'train.csv')
run(dir + 'test.csv')

  
