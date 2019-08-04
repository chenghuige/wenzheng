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

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

from dataset import *
from model import *
import model as base
import evaluate as ev
import loss

def main(_):
  melt.init()
  fit = melt.get_fit()

  FLAGS.eval_batch_size = 512 * FLAGS.valid_multiplier
  print('---------eval_batch_size', FLAGS.eval_batch_size)

  model_name = FLAGS.model
  model = getattr(base, model_name)() 

  Dataset = TextDataset if not 'tfrecord' in FLAGS.train_input else TFRecordDataset

  loss_fn = tf.losses.sigmoid_cross_entropy if not FLAGS.rank_loss else loss.binary_crossentropy_with_ranking

  fit(model,  
      loss_fn,
      Dataset,
      eval_fn=ev.evaluate,
      valid_write_fn=ev.valid_write,
      write_valid=FLAGS.write_valid)   

if __name__ == '__main__':
  tf.app.run()  
