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

import sys,os
import numpy as np
import gezi

import json 
import traceback

START_WORD = '<S>'
END_WORD = '</S>'

num = 0
num_errs = 0
for line in sys.stdin:
  line = line.rstrip()
  # try:
  m = json.loads(line)
  m['question'] = gezi.to_simplify(m['question'])
  for i in range(len(m['answers'])):
    m['answers'][i] = gezi.to_simplify(m['answers'][i])
  for i in range(len(m['documents'])):
    m['documents'][i]['title'] = gezi.to_simplify(m['documents'][i]['title'])
    for j in range(len(m['documents'][i]['paragraphs'])):
      m['documents'][i]['paragraphs'][j] = gezi.to_simplify(m['documents'][i]['paragraphs'][j])
  print(json.dumps(m, ensure_ascii=False).encode('utf8'))

