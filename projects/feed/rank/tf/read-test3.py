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
#eager模式因为非repeat 所以最后才会出现small batch 变成补0
#如果是repeat模式不会的 repeat 模式 训练会比较安全。。 完全按照step来。。 TODO eager模式 当前框架按照非repeate 最后一组有风险 对应parse batch模式 CHECK
#而且非repeate模式 如果每个epoch seed相同数据顺序是完全一致的 repeat没关系

# notcie input data 10 lines

def main(_):
  FLAGS.train_input = '../input/train.small'
  FLAGS.valid_input = '../input/train.small'
  FLAGS.batch_size = 4
  FLAGS.feat_file_path='../input/feature_index'
  FLAGS.field_file_path='../input/feat_fields.old'
  melt.init()

  #dataset = Dataset('train')
  dataset = Dataset('valid')
  
  iter = dataset.make_batch()
  op = iter.get_next()

  print('---batch_size', dataset.batch_size, FLAGS.batch_size)  

  sess = melt.get_session()

  print('----sess', sess)
  try:
    sess.run(iter.initializer)
  except Exception:
    pass

  if not FLAGS.use_horovod:
    #for epoch in range(2):
    #  for i in range(int(10 / FLAGS.batch_size + 0.5)):
    #    batch = sess.run(op)
    #    print(epoch, i, len(batch[0]['id']))
    for i in range(4):
      batch = sess.run(op)
      print(i, batch[0]['id'])
  else:
    for epoch in range(1):
      for i in range(3):
        batch = sess.run(op)
        print(epoch, i, len(batch[0]['id']))

if __name__ == '__main__':
  tf.app.run()  
