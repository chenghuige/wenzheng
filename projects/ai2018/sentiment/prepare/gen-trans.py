#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-trans.py
#        \author   chenghuige  
#          \date   2018-10-03 22:43:06.305917
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

#flags.DEFINE_string('input', '', '')
#flags.DEFINE_string('output', '', '')

import sys 
import os

import pandas as pd
from tqdm import tqdm

df = pd.read_csv('./train.csv')

comments = {}
for line in open('./train.en.txt'):
  try:
    id, comment = line.rstrip().split('\t')
    comments[id] = comment
  except Exception:
    pass

  
print(len(comments))

num_modify = 0
for i in tqdm(range(len(df)), ascii=True):
  row = df.iloc[i]
  id = row['id']
  id = str(id)
  if id in comments:
    df.loc[i,'content'] = comments[id]
    num_modify += 1
  
print(num_modify)
df.to_csv('./trans.en.csv', index=False, encoding="utf_8_sig")

def main(_):
  pass

if __name__ == '__main__':
  tf.app.run()  
  
