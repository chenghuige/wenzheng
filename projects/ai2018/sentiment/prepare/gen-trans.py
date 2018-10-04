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
num_bads = 0
for i in tqdm(range(len(df)), ascii=True):
  row = df.iloc[i]
  id = row['id']
  id = str(id)
  if id in comments: 
    if len(comments[id].replace(' ', '')) < len(row['content'].replace(' ', '')) * 1.5 and len(comments[id].replace(' ', '')) > len(row['content'].replace(' ', '')) * 0.7:
      df.loc[i,'content'] = comments[id]
      num_modify += 1
    else:
      print('bad translate:', comments[id])
      print('ori', row['content'])
      num_bads += 1
  
print(num_modify, num_bads)
df.to_csv('./trans.en.csv', index=False, encoding="utf_8_sig")

def main(_):
  pass

if __name__ == '__main__':
  tf.app.run()  
  
