#!/usr/bin/env python
#encoding=utf8
# ==============================================================================
#          \file   to_flickr_caption.py
#        \author   chenghuige  
#          \date   2016-07-11 16:29:27.084402
#   \Description   seg using dianping corpus /home/gezi/data/ai2018/sentiment/dianping/ratings.csv 
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('seg_method', 'basic', '')
flags.DEFINE_string('name', None, '')
flags.DEFINE_bool('for_pretrain', False, '')
flags.DEFINE_string('sp_path', None, '')

assert FLAGS.seg_method

import sys,os
import numpy as np

import gezi

import pandas as pd

from tqdm import tqdm
import traceback

from projects.ai2018.sentiment.prepare import filter
from third.bert import tokenization

tokenizer = tokenization.BasicTokenizer()

def seg(id, text, out):
  text = filter.filter(text)
  words = tokenizer.tokenize(text)
  print(id, '\x09'.join(words), sep='\t', file=out)

assert FLAGS.name
ifile = sys.argv[1]
ofile = ifile.replace('.csv', '.seg.%s.txt' % FLAGS.name)

num_errs = 0
with open(ofile, 'w') as out:
  df = pd.read_csv(ifile, lineterminator='\n')
  contents = df['content'].values 
  ids = df['id'].values
  for i in tqdm(range(len(df)), ascii=True):
    try:
      seg(ids[i], contents[i], out)
    except Exception:
      if num_errs == 0:
        print(traceback.format_exc())
      num_errs += 1
      continue
    #exit(0)

print('num_errs:', num_errs, 'ratio:', num_errs / len(df))
