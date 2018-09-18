#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   ensemble.py
#        \author   chenghuige  
#          \date   2018-09-15 19:04:21.026718
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import glob
import pandas as pd
import numpy as np 
import gezi

idir = sys.argv[1]


def parse(input):
  return np.array([float(x.strip()) for x in input[1:-1].split(' ') if x.strip()])

cadidates = {}
results = {}
results2 = {}
for file_ in glob.glob('%s/*.infer.txt.debug' % idir):
  df = pd.read_csv(file_)
  #df = df.sort_index(0)
  ids = df['id'].values
  candidates_ = df['candidates'].values
  scores = df['score']
  scores = [parse(score) for score in scores]

  if not results:
    candidates = dict(zip(ids, candidates_))

  print(file_)

  scores =  gezi.softmax(scores, -1)
  for id, score in zip(ids, scores):
    if id not in results:
      results[id] = score 
    else:
      results[id] += score

  # scores2 = gezi.softmax(scores, -1)
  # for id, score in zip(ids, scores2):
  #   if id not in results2:
  #     results2[id] = score 
  #   else:
  #     results2[id] += score

ofile = os.path.join(idir, 'ensemble.infer.txt')
print('ofile:', ofile)
with open(ofile, 'w') as out:
  for id, score in results.items():
    index = np.argmax(score, -1)
    #print(id, score, index)
    predict = candidates[id].split('|')[index]
    print(id, predict, sep='\t', file=out)



