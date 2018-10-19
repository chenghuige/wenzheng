#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   melt_light.py
#        \author   chenghuige  
#          \date   2018-10-17 13:24:31.571379
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import six

def int_feature(value):
  import tensorflow as tf
  if not isinstance(value, (list,tuple)):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def int64_feature(value):
  import tensorflow as tf
  if not isinstance(value, (list,tuple)):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  import tensorflow as tf
  if not isinstance(value, (list,tuple)):
    value = [value]
  if not six.PY2:
    if isinstance(value[0], str):
      value = [x.encode() for x in value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature(value):
  import tensorflow as tf
  if not isinstance(value, (list,tuple)):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

features = lambda d: tf.train.Features(feature=d)

# Helpers for creating SequenceExample objects  copy from \tensorflow\python\kernel_tests\parsing_ops_test.py
feature_list = lambda l: tf.train.FeatureList(feature=l)
feature_lists = lambda d: tf.train.FeatureLists(feature_list=d)


def int64_feature_list(values):
  import tensorflow as tf
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[int64_feature(v) for v in values])

def bytes_feature_list(values):
  import tensorflow as tf
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[bytes_feature(v) for v in values])

def float_feature_list(values):
  import tensorflow as tf
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[float_feature(v) for v in values])

