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
tf.enable_eager_execution()

import melt

def main(_):
  FLAGS.train_input = '../input/train.small'
  FLAGS.valid_input = '../input/train.small'
  FLAGS.batch_size = 4
  FLAGS.feat_file_path='../input/feature_index'
  FLAGS.field_file_path='../input/feat_fields.old'
  #melt.init()

  dataset = Dataset('train')
  #dataset = Dataset('valid')

  da = dataset.make_batch(4)

  # #as for melt eager not repeat mode will get same sequence each epoch if seed is the same
  # for epoch in range(2):
  #   for i, batch in enumerate(da):
  #     print(i, batch[1], batch[0]['id'])
      
  for i, batch in enumerate(da):
      print('---------------------------', i)
      #print(batch[0]['id'])
      print(batch[0]['id'], batch[0]['id'].shape, batch[1].shape, batch[0]['index'].shape)


if __name__ == '__main__':
  tf.app.run()  
