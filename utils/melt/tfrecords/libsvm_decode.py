#!/usr/bin/env python
# ==============================================================================
#          \file   libsvm_decode.py
#        \author   chenghuige  
#          \date   2016-08-15 20:17:53.507796
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

#notice heare use parse_example not parse single example for it
#is used in sparse record reading flow, shuffle then deocde
def decode(batch_serialized_examples, label_type=tf.int64, index_type=tf.int64, value_type=tf.float32):
  """
  decode batch_serialized_examples for use in parse libsvm fomrat sparse tf-record
  Returns:
  X,y
  """
  features = tf.parse_example(
      batch_serialized_examples,
      features={
          'label' : tf.FixedLenFeature([], label_type),
          'index' : tf.VarLenFeature(index_type),
          'value' : tf.VarLenFeature(value_type),
      })

  label = features['label']
  index = features['index']
  value = features['value']

  #return as X,y
  print(index, value)
  return (index, value), label
