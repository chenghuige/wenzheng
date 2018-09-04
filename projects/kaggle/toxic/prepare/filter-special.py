#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   filter-specail.py
#        \author   chenghuige  
#          \date   2018-03-21 01:43:34.617574
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import pandas as pd

pdf = pd.read_csv('/home/gezi/temp/toxic/v16/tfrecords/glove.lower/ensemble_0359_124678_lc_handled.csv')
ids = pdf['id'].values
toxics = pdf['toxic'].values

m = {}
for id, toxic  in zip(ids, toxics):
  m[id] = toxic

df = pd.read_csv('/home/gezi/temp/toxic/v16/tfrecords/glove.lower/test.special.csv')

ids = df['id'].values
comments = df['comment_text'].values 
toxics = []

for id, comment in zip(ids, comments):
  toxics.append(m[id])

df['toxic'] = toxics 

df = df.sort_values(['toxic'], ascending=[0])

df.to_csv('/home/gezi/temp/toxic/v16/tfrecords/glove.lower/test.special.score.csv', index=False)

