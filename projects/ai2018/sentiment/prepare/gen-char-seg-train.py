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

flags.DEFINE_string('seg_method', 'char', '')
flags.DEFINE_integer("max_lines", 0, "")

assert FLAGS.seg_method

import sys,os
import numpy as np
import melt

from gezi import Segmentor
segmentor = Segmentor()

import gezi

import pandas as pd

from projects.ai2018.sentiment.prepare import filter

START_WORD = '<S>'
END_WORD = '</S>'

print('seg_method:', FLAGS.seg_method, file=sys.stderr)

def seg(text, out):
  text = filter.filter(text)
  words = segmentor.Segment(text, FLAGS.seg_method)
  words = [x.strip() for x in words if x.strip()]
  if words:
    print(' '.join(words), file=out)

ifiles = ['/home/gezi/data/ai2018/sentiment/train.csv',
          '/home/gezi/data/ai2018/sentiment/valid.csv',
          '/home/gezi/data/ai2018/sentiment/test.csv']

ofile = '/home/gezi/data/ai2018/sentiment/seg.char.txt'

with open(ofile, 'w') as out:
  num = 0
  for ifile in ifiles:
    df = pd.read_csv(ifile)
    for comment in df['content']:
      if num % 10000 == 0:
        print(num, file=sys.stderr)
      try:
        seg(comment, out)
      except Exception:
        continue
      num += 1
      if num == FLAGS.max_lines:
        break

