#!/usr/bin/env python
# ==============================================================================
#          \file   sparse_ops.py
#        \author   chenghuige  
#          \date   2016-08-16 10:09:41.241790
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
  
def sparse_tensor_to_dense(input_tensor, maxlen=0):
  """
  notice maxlen must > your max real index
  otherwise runtime check error like 
  Invalid argument: indices[3] = [0,3] is out of bounds: need 0 <= index < [5,3]
  @FIXME still face this might be tf bug, when running mutlitple tf reading same data ?
  """
  if maxlen <= 0:
    return tf.sparse_tensor_to_dense(input_tensor)
  else:
    return tf.sparse_to_dense(input_tensor.indices, 
                              [input_tensor.dense_shape[0], maxlen], 
                              input_tensor.values)