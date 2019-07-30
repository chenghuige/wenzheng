#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2019-07-26 23:00:24.215922
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

import melt 
import numpy as np

from config import *

class Dataset(melt.tfrecords.Dataset):
  def __init__(self, subset='valid'):
    super(Dataset, self).__init__(subset)
    # only support line parse not batch parse since here parse_single_example
    # so slower then text + pyfunc due to they use batch parse
  
  def parse(self, example):
    features_dict = {
      'id':  tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64),
      'index': tf.VarLenFeature(tf.int64),
      'field': tf.VarLenFeature(tf.int64),
      'value': tf.VarLenFeature(tf.float32),
      }

    features = tf.parse_single_example(example, features=features_dict)

    print(features)

    melt.sparse2dense(features)

    y = features['label']
    y = tf.cast(y, tf.float32)
    del features['label']

    x = features

    return x, y
