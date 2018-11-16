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

import six
if FLAGS.seg_method == 'char':
  assert not six.PY2

import sys,os
import numpy as np

import gezi

#assert gezi.env_has('JIEBA_POS')
from gezi import WordCounter 

import pandas as pd

from projects.ai2018.sentiment.prepare import filter

from tqdm import tqdm
import traceback

START_WORD = '<S>'
END_WORD = '</S>'

counter = WordCounter(most_common=0, min_count=1)
counter2 = WordCounter(most_common=0, min_count=1)

print('seg_method:', FLAGS.seg_method, file=sys.stderr)

if gezi.env_has('SENTENCE_PIECE'):
  assert FLAGS.sp_path 
  gezi.segment.init_sp(FLAGS.sp_path)

def seg(id, text, out, type):
  text = filter.filter(text)
  counter.add(START_WORD)
  counter.add(END_WORD)
  l = gezi.cut(text, type)

  if type != 'word':
    for x, y in l:
      counter.add(x)
      counter2.add(y)
    words = ['%s|%s' % (x, y) for x,y in l]
  else:
    if FLAGS.seg_method == 'char':
      l2 = []
      for i, w in enumerate(l):
        for ch in w:
          counter.add(ch)
          counter2.add(str(i))
          l2.append((ch, i))
      words =  ['%s|%d' % (x, y) for x,y in l2]
    else:
      words = l
      for w in words:
        counter.add(w)

  if not FLAGS.for_pretrain:
    print(id, '\x09'.join(words), sep='\t', file=out)
  else:
    print(' '.join([x.split('|')[0] for x in words]), file=out)

assert FLAGS.name
ifile = sys.argv[1]
ofile = ifile.replace('.csv', '.seg.%s.txt' % FLAGS.name)
vocab = ifile.replace('.csv', '.seg.%s.vocab' % FLAGS.name)
type_ = 'word'

vocab2 = None
if 'pos' in FLAGS.name or FLAGS.seg_method == 'char':
  vocab2 = ifile.replace('.csv', '.pos.%s.vocab' % FLAGS.name)
  if 'pos' in FLAGS.name:
    type_ = 'pos'
elif 'ner' in FLAGS.name:
  vocab2 = ifile.replace('.csv', '.ner.%s.vocab' % FLAGS.name)
  type_ = 'ner'

num_errs = 0
with open(ofile, 'w') as out:
  df = pd.read_csv(ifile, lineterminator='\n')
  contents = df['content'].values 
  ids = df['id'].values
  for i in tqdm(range(len(df)), ascii=True):
    #if str(ids[i]) in ids_set:
    #  continue
    #if i != 2333:
    #  continue
    #print(gezi.cut(filter.filter(contents[i]), type_))
    try:
      seg(ids[i], contents[i], out, type=type_)
    except Exception:
      if num_errs == 0:
        print(traceback.format_exc())
      num_errs += 1
      continue
    #exit(0)

counter.save(vocab)
if vocab2:
  counter2.save(vocab2)

print('num_errs:', num_errs, 'ratio:', num_errs / len(df))
