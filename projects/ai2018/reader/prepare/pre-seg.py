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
import json

START_WORD = '<S>'
END_WORD = '</S>'

counter = WordCounter(most_common=0, min_count=1)
counter2 = WordCounter(most_common=0, min_count=1)

print('seg_method:', FLAGS.seg_method, file=sys.stderr)

type = None

if gezi.env_has('SENTENCE_PIECE'):
  assert FLAGS.sp_path 
  gezi.segment.init_sp(FLAGS.sp_path)

def seg_(text):
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

  return words

def seg(m, out):
  # query
  counter.add(START_WORD)
  counter.add(END_WORD)
  # passage
  counter.add(START_WORD)
  counter.add(END_WORD)

  words = seg_(m['query'])
  m['seg_query'] = '\x09'.join(words)
  words = seg_(m['passage'])
  m['seg_passage'] = '\x09'.join(words)

  l = []
  for x in m['alternatives'].split('|'):
    words = seg_(x)
    l.append('\x09'.join(words))
  m['seg_alternatives'] = '|'.join(l)

  if six.PY2:
    print(json.dumps(m, ensure_ascii=False).encode('utf8'), file=out)
  else:
    print(json.dumps(m, ensure_ascii=False), file=out)

assert FLAGS.name
ifile = sys.argv[1]
ofile = ifile.replace('.json', '.seg.%s.json' % FLAGS.name)
vocab = ifile.replace('.json', '.seg.%s.vocab' % FLAGS.name)
type = 'word'

vocab2 = None
if 'pos' in FLAGS.name or FLAGS.seg_method == 'char':
  vocab2 = ifile.replace('.json', '.pos.%s.vocab' % FLAGS.name)
  if 'pos' in FLAGS.name:
    type = 'pos'
elif 'ner' in FLAGS.name:
  vocab2 = ifile.replace('.json', '.ner.%s.vocab' % FLAGS.name)
  type = 'ner'

fm = 'w'
ids_set = set()
# if os.path.exists(ofile):
#   fm = 'a'
#   for line in open(ofile):
#     m = json.loads(line.rstrip('\n'))
#     ids_set.add(m['query_id'])

print('%s already done %d' % (ofile, len(ids_set)))

num_errs = 0
with open(ofile, fm) as out:
  lines = open(ifile).readlines()
  for i in tqdm(range(len(lines)), ascii=True):
    m = json.loads(lines[i].rstrip('\n'))
    id = m['query_id']
    if id in ids_set:
      continue
    try:
      seg(m, out)
    except Exception:
      if num_errs == 0:
        print(traceback.format_exc())
      num_errs += 1
      continue
    #exit(0)

counter.save(vocab)
if vocab2:
  counter2.save(vocab2)

print('num_errs:', num_errs, 'ratio:', num_errs / len(lines))
