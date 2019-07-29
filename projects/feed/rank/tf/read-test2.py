#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   read.py
#        \author   chenghuige  
#          \date   2019-07-26 18:02:22.038876
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from text_dataset import Dataset

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

import melt

#非eager模式 melt是按照step控制轮次的 使用的是 repeat 模式
#eager模式 melt 非repeate

# notcie input data 10 lines

def main(_):
  melt.init()

  FLAGS.train_input = '../input/train.small'
  FLAGS.valid_input = '../input/train.small'
  FLAGS.batch_size = 5
  FLAGS.feat_file_path='../input/feature_index'
  FLAGS.field_file_path='../input/feat_fields.old'

  dataset = Dataset('train')
  #dataset = Dataset('valid')

  iter = dataset.make_batch()
  op = iter.get_next()
  
  sess = melt.get_session()

  print('----sess', sess)

  if not FLAGS.use_horovod:
    for epoch in range(2):
      for i in range(int(10 / FLAGS.batch_size + 0.5)):
        batch = sess.run(op)
        print(epoch, i, batch[0]['id'])
  else:
    for epoch in range(1):
      for i in range(1):
        batch = sess.run(op)
        print(epoch, i, batch[0]['id'])

if __name__ == '__main__':
  tf.app.run()  