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

#flags.DEFINE_string('seg_method', 'basic', '')
flags.DEFINE_integer("max_lines", 0, "")

#assert FLAGS.seg_method

import sys,os
import numpy as np
import melt

from gezi import Segmentor
segmentor = Segmentor()

import gezi

import pandas as pd 

from wenzheng.utils import text2ids

vocab = '/home/gezi/temp/ai2018/sentiment/vocab.5k.chars.txt'
text2ids.init(vocab)

from text2ids import text2ids as text2ids_ 

#import filter

START_WORD = '<S>'
END_WORD = '</S>'

print('seg_method:', FLAGS.seg_method, file=sys.stderr)
def seg(text, out):
  #text = filter.filter(text)
  words = text2ids.ids2words(text2ids_(text)) 
  words = [x.strip() for x in words if x.strip()]
  if words:
    print(' '.join(words), file=out)

ifile = '/home/gezi/data/ai2018/sentiment/sentiment_classify_data/comment_raw_v2/raw_comment_v2.csv'
df = pd.read_csv(ifile)

ofile = '/home/gezi/data/ai2018/sentiment/sentiment_classify_data/seg.mix.txt'

with open(ofile, 'w') as out:
  num = 0
  for comment in df['content']:
    if num % 10000 == 0:
      print(num, file=sys.stderr)
    #try:
    seg(comment, out)
    #except Exception:
    #  continue
    num += 1
    if num == FLAGS.max_lines:
      break

