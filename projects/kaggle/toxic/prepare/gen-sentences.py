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

gezi.segment.init_spacy_full(['ner', 'tagger'])

num_threads = 12

input = None
def run(index):
  df = pd.read_csv(input)
  ids = df['id'].values
  comments = df['comment_text'].values 

  start, end = gezi.get_fold(len(comments), num_threads, index)
  
  output = input.replace('.csv', '.sents.%d.txt' % index) 
  print(output)
  num = 0
  with open(output, 'w') as out:
    for id, comment in zip(ids[start:end], comments[start:end]):
      if num % 1000 == 0:
        print(num)
      num += 1
      doc = gezi.segment.doc(comment)
      for sent in doc.sents:
        print(id, sent.text.replace('\n', 'NEWLINE'), sep='\t', file=out)


if sys.argv[1] == 'train':
  input = '/home/gezi/data/kaggle/toxic/train.csv'
else:
  input = '/home/gezi/data/kaggle/toxic/test.csv'


pool = mp.Pool()
pool.map(run, range(num_threads))
pool.close()
pool.join()

