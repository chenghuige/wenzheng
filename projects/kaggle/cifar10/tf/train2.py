#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2018-01-13 16:32:26.966279
#   \Description  
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

#flags.DEFINE_string('model_dir', './mount/temp/cifar10/model/resnet', '')
flags.DEFINE_string('algo', 'resnet', '')

import numpy as np

import melt 
logging = melt.logging
import gezi
import traceback

import cifar10
from cifar10_model import Model

from loss import criterion
import evaluate as ev

def get_dataset(subset):
  data_dir = './mount/data/cifar10/' 
  use_distortion = False
  if subset == 'train':
    use_distortion = True
  return cifar10.Cifar10DataSet(data_dir, subset=subset, use_distortion=use_distortion) 


# TODO FIXME why 1gpu ok 2gpu fail ? train.py all ok... ai2018/sentiment/model.py bert is also all ok why here wrong ?
def main(_):
  melt.apps.init()

  model = Model()

  logging.info(model)
  fit = melt.apps.get_fit()

  fit(get_dataset,
      model,  
      criterion,
      eval_fn=ev.evaluate,
      valid_write_fn=ev.valid_write,
      infer_write_fn=ev.infer_write,
      valid_suffix='.valid.csv',
      infer_suffix='.infer.csv')   

if __name__ == '__main__':
  tf.app.run()  
