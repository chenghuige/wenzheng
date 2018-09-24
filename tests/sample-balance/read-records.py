#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   inference.py
#        \author   chenghuige  
#          \date   2018-02-05 20:05:25.123740
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', '/tmp/ai2018/reader/tfrecord/valid/*record,', '')
flags.DEFINE_integer('batch_size_', 50, '')
flags.DEFINE_string('type', 'debug', '')
flags.DEFINE_string('base', './mount/temp/ai2018/reader/tfrecord/', '')
#flags.DEFINE_integer('fold', None, '')

import tensorflow as tf
tf.enable_eager_execution()

import sys, os
from sklearn import metrics
import pandas as pd 
import numpy as np
import gezi


import pickle

from wenzheng.utils import ids2text

import melt
logging = melt.logging
from dataset import Dataset

# random seed works
# python read-records.py  --random_seed=1024

def main(_):
  base = FLAGS.base
  logging.set_logging_path('./mount/tmp/')
  vocab_path = os.path.join(os.path.dirname(os.path.dirname(FLAGS.input)), 'vocab.txt')
  ids2text.init(vocab_path)
  FLAGS.vocab = f'{base}/vocab.txt'

  tf.set_random_seed(FLAGS.random_seed)

  # FLAGS.length_index = 2
  # FLAGS.buckets = '100,400'
  # FLAGS.batch_sizes = '64,64,32'

  input_ = FLAGS.input 
  if FLAGS.type == 'test':
    input_ = input_.replace('valid', 'test')

  inputs = gezi.list_files(input_)
  inputs.sort()
  if FLAGS.fold is not None:
    inputs = [x for x in inputs if not x.endswith('%d.record' % FLAGS.fold)]

  print('type', FLAGS.type, 'inputs', inputs, file=sys.stderr)

  #dataset = Dataset('valid')
  dataset = Dataset('train')

  # balance pos neg tested ok
  dataset = dataset.make_batch(FLAGS.batch_size_, inputs, repeat=False)

  print('dataset', dataset)

  ids = []

  timer = gezi.Timer('read record')
  for i, (x, y) in enumerate(dataset):
    #if i % 10 == 1:
    #  print(x['passage'][0])
    #  print(ids2text.ids2text(x['passage'][0], sep='|'))
    #  print(ids2text.ids2text(x['candidate_pos'][0], sep='|'))
    #  print(ids2text.ids2text(x['candidate_neg'][0], sep='|'))
    #  print(x['passage'])
    #  print(x['candidate_pos'])
    #  print(type(x['id'].numpy()[0]) == bytes)
    #  break 
    for id in x['id'].numpy():
      ids.append(id)
    print(i, x['type'].numpy())

  print(len(ids), len(set(ids)))
  #print(ids)

if __name__ == '__main__':
  tf.app.run()
