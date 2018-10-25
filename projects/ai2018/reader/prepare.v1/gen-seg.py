#!/usr/bin/env python
#encoding=utf8
# ==============================================================================
#          \file   to_flickr_caption.py
#        \author   chenghuige  
#          \date   2016-07-11 16:29:27.084402
#   \Description   seg using dureader corpus /home/gezi/data/dureader/raw/trainset/*.json
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('seg_method', 'basic', '')
flags.DEFINE_integer("max_lines", 0, "")

assert FLAGS.seg_method

import sys,os
import numpy as np
import melt

from gezi import Segmentor
segmentor = Segmentor()

import gezi

import json

START_WORD = '<S>'
END_WORD = '</S>'

print('seg_method:', FLAGS.seg_method, file=sys.stderr)

def seg(text):
  words = segmentor.Segment(text, FLAGS.seg_method)
  words = [x.strip() for x in words if x.strip()]
  print(' '.join(words))

num = 0
for line in sys.stdin:
  if num % 10000 == 0:
    print(num, file=sys.stderr)
  line = line.rstrip()
  m = json.loads(line)
  question = m['question']
  seg(question)
  answers = m['answers']
  for answer in answers:
    seg(answer)
  docs = m['documents']
  for doc in docs:
    title = doc['title']
    seg(title)
    for paragrah in doc['paragraphs']:
      seg(paragrah)
  num += 1
  if num == FLAGS.max_lines:
    break

