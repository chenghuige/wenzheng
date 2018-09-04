#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   post-deal.py
#        \author   chenghuige  
#          \date   2018-03-21 05:29:51.134440
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import pandas as pd 
import numpy as np

white_csv = pd.read_csv(sys.argv[2])

white_ids = set(white_csv['id'].values)

black_csv = pd.read_csv(sys.argv[3])
black_ids = set(black_csv['id'].values) 


assert len(white_ids) > len(black_ids)

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
df = pd.read_csv(sys.argv[1])


new_scores = []
ids = df['id'].values
scores = df[CLASSES].values 

for id, score in zip(ids, scores):
  score = list(score)
  if id in white_ids:
    score = np.array([x * 0.001 for x in score])
  elif id in black_ids:
    score[0] = 1.0
    score = np.array(score)
  new_scores.append(score)

df[CLASSES] = new_scores

df.to_csv(sys.argv[4], index=False)
