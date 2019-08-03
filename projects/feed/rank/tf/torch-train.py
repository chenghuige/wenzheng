#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   torch-train.py
#        \author   chenghuige  
#          \date   2019-08-02 01:05:59.741965
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

import torch
from torch import nn
from torch.nn import functional as F

from dataset import *
from pyt.model import *
import pyt.model as base
import evaluate as ev
import loss

import melt

def main(_):
  FLAGS.torch = True
  melt.apps.init()
  fit = melt.apps.get_fit()

  FLAGS.eval_batch_size = 512 * FLAGS.valid_multiplier
  print('---------eval_batch_size', FLAGS.eval_batch_size)

  model_name = FLAGS.model
  model = getattr(base, model_name)() 

  Dataset = TextDataset if not 'tfrecord' in FLAGS.train_input else TFRecordDataset

  loss_fn = nn.BCEWithLogitsLoss()

  fit(Dataset,
      model,  
      loss_fn,
      eval_fn=ev.evaluate,
      valid_write_fn=ev.valid_write,
      write_valid=FLAGS.write_valid)   

if __name__ == '__main__':
  tf.app.run()  
  
