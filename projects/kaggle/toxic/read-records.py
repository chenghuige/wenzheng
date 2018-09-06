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

flags.DEFINE_string('input', '/home/gezi/mount/temp/kaggle/toxic/tfrecords/glove/test/*record,', '')
flags.DEFINE_integer('batch_size_', 512, '')
flags.DEFINE_string('model_dir', None, '')
flags.DEFINE_string('type', 'train', '')
#flags.DEFINE_integer('fold', None, '')

import tensorflow as tf
tf.enable_eager_execution()

import sys, os
from sklearn import metrics
import pandas as pd 
import numpy as np
import gezi

from wenzheng.utils import ids2text

import melt
logging = melt.logging
from dataset import Input, Dataset

def main(_):
  logging.set_logging_path('/home/gezi/tmp/')
  vocab_path = os.path.join(os.path.dirname(os.path.dirname(FLAGS.input)), 'vocab.txt')
  ids2text.init(vocab_path)
  FLAGS.vocab = '/home/gezi/mount/temp/kaggle/toxic/tfrecords/glove/vocab.txt'

  FLAGS.length_index = 2
  #FLAGS.length_index = 1
  FLAGS.buckets = '100,400'
  FLAGS.batch_sizes = '64,64,32'

  input_ = FLAGS.input 
  if FLAGS.type == 'test':
    input_ = input_.replace('train', 'test')

  inputs = gezi.list_files(input_)
  inputs.sort()
  if FLAGS.fold is not None:
    inputs = [x for x in inputs if not x.endswith('%d.record' % FLAGS.fold)]

  print('type', FLAGS.type, 'inputs', inputs, file=sys.stderr)

  dataset = Dataset('valid')
  dataset = dataset.make_batch(FLAGS.batch_size_, inputs)

  print('dataset', dataset)

  timer = gezi.Timer('read record')
  for i, input in enumerate(dataset):
    if i % 10 == 1:
      #print(i, result)
      #input = Input(*input)
      print(input.classes)
      print(input.comment)
      print(ids2text.ids2text(input.comment[0], sep='|'))
      break


if __name__ == '__main__':
  tf.app.run()
