#!/usr/bin/env python
# ==============================================================================
#          \file   ops.py
#        \author   chenghuige  
#          \date   2016-08-16 10:09:34.992292
#   \Description  
# ==============================================================================

"""
TODO
tf.sign should be replaced by tf.unequal 0, to allow -1
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
# 'Constant' gets imported in the module 'array_ops'.
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_array_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.deprecation import deprecated
# pylint: enable=wildcard-import


#as tf_utils in order to avoid confilict/hidden melt.utils since (from melt.ops import * will import..)
#@NOTICE! be careful or if import uitls without rename, you need to from melt.ops not import * and use melt.ops..
from tensorflow.contrib.layers.python.layers import utils as tf_utils

import numpy as np

# TODO FIXME.. hurt.. WHY ? maybe due to dataset pipline...
#def greater_then_set(x, thre, val):
  #mask = tf.cast(x > thre, dtype=x.dtype)
  #rmask = 1 - mask
  #return x * rmask + mask * val
  #return x

#https://stackoverflow.com/questions/39921607/how-to-make-a-custom-activation-function-with-only-python-in-tensorflow
def greater_then_set(x, thre, val):
   cond = tf.greater(x, thre)
   return tf.where(cond, tf.ones_like(x, dtype=x.dtype) * val, x)
   #return tf.where(cond, tf.constant(x, dtype=x.dtype, shape=x.shape) * val, x)
  #return x

def matmul(X, w):
  """ General matmul  that will deal both for dense and sparse input
  hide the differnce of dense and sparse input for end users
  Since sparse usage is much less then dense, try to use slim. or tf.contrib.layers directly if possible
  https://github.com/tensorflow/tensorflow/issues/342
  Args:
  X: a tensor, or a list with two sparse tensors (index, value)
  w: a tensor
  """
  if isinstance(X, tf.Tensor):
    return tf.matmul(X,w)
  else: 
    #X[0] index, X[1] value
    return tf.nn.embedding_lookup_sparse(w, X[0], X[1], combiner='sum')

#@TODO try to use slim.fully_connected
def mlp_forward(input, hidden, hidden_bais, out, out_bias, activation=tf.nn.relu, name=None):
  #@TODO **args?
  with tf.name_scope(name, 'mlp_forward', [input, hidden, hidden_bais, out, out_bias, activation]):
    hidden_output = activation(matmul(input, hidden) + hidden_bais)
    return tf.matmul(hidden_output, out) + out_bias

def mlp_forward_nobias(input, hidden, out, activation=tf.nn.relu, name=None):
  with tf.name_scope(name, 'mlp_forward_nobias', [input, hidden, out, activation]):
    hidden_output = activation(matmul(input, hidden))
    return tf.matmul(hidden_output, out) 

def element_wise_dot(a, b, keep_dims=True, name=None):
  with tf.name_scope(name, 'element_wise_cosine_nonorm', [a, b]):
    return tf.reduce_sum(tf.multiply(a, b), -1, keep_dims=keep_dims)

def element_wise_cosine_nonorm(a, b, keep_dims=True, name=None):
  with tf.name_scope(name, 'element_wise_cosine_nonorm', [a, b]):
    return tf.reduce_sum(tf.multiply(a, b), -1, keep_dims=keep_dims)

#[batch_size, y], [batch_size, y] => [batch_size, 1]
def element_wise_cosine(a, b, a_normed=False, b_normed=False, nonorm=False, keep_dims=True, name=None):
  if nonorm:
    return element_wise_cosine_nonorm(a, b, keep_dims, name)
  with tf.name_scope(name, 'element_wise_cosine', [a,b]):
    if a_normed:
      normalized_a = a 
    else:
      normalized_a = tf.nn.l2_normalize(a, -1)
    if b_normed:
      normalized_b = b 
    else:
      normalized_b = tf.nn.l2_normalize(b, -1)
    #return tf.matmul(normalized_a, normalized_b, transpose_b=True)
    return tf.reduce_sum(tf.multiply(normalized_a, normalized_b), -1, keep_dims=keep_dims)

def dot(a, b, name=None):
  with tf.name_scope(name, 'dot', [a,b]):
    return tf.matmul(a, b, transpose_b=True)   

#actually is dot..
def cosine_nonorm(a, b, name=None):
  with tf.name_scope(name, 'cosine_nonorm', [a,b]):
    return tf.matmul(a, b, transpose_b=True)  

#[batch_size, y] [x, y] => [batch_size, x]
def cosine(a, b, a_normed=False, b_normed=False, nonorm=False, name=None):
  if nonorm:
    return cosine_nonorm(a, b)
  with tf.name_scope(name, 'cosine', [a,b]):
    if a_normed:
      normalized_a = a 
    else:
      normalized_a = tf.nn.l2_normalize(a, -1)
    if b_normed:
      normalized_b = b 
    else:
      normalized_b = tf.nn.l2_normalize(b, -1)
    return tf.matmul(normalized_a, normalized_b, transpose_b=True)

def reduce_mean(input_tensor,  reduction_indices=None, keep_dims=False):
  """
  reduce mean with mask
  """
  return tf.reduce_sum(input_tensor, reduction_indices=reduction_indices, keep_dims=keep_dims) / \
         tf.reduce_sum(tf.sign(input_tensor), reduction_indices=reduction_indices, keep_dims=keep_dims)

def masked_reduce_mean(input_tensor,  reduction_indices=None, keep_dims=False, mask=None):
  """
  reduce mean with mask
  [1,2,3,0] -> 2 not 1.5 as normal
  """
  if mask is None:
    mask = tf.sign(input_tensor)
  return tf.reduce_sum(input_tensor, reduction_indices=reduction_indices, keep_dims=keep_dims) / \
         tf.reduce_sum(mask, reduction_indices=reduction_indices, keep_dims=keep_dims)

def reduce_mean_with_mask(input_tensor, mask, reduction_indices=None, keep_dims=False):
  return  tf.reduce_sum(input_tensor, reduction_indices=reduction_indices, keep_dims=keep_dims) / \
          tf.reduce_sum(mask, reduction_indices=reduction_indices, keep_dims=keep_dims)

def embedding_lookup(emb, index, reduction_indices=None, combiner='mean', name=None):
  with tf.name_scope(name, 'emb_lookup_%s'%combiner, [emb, index]):
    lookup_result = tf.nn.embedding_lookup(emb, index)
    if combiner == 'mean':
      return tf.reduce_mean(lookup_result, reduction_indices)
    elif combiner == 'sum':
      return tf.reduce_sum(lookup_result, reduction_indices)
    else:
      raise ValueError('Unsupported combiner: ', combiner)

# TODO this mean is meaning less.. FIXME should be / real_lenght
def embedding_lookup_mean(emb, index, reduction_indices=None, name=None):
  with tf.name_scope(name, 'emb_lookup_mean', [emb, index]):
    lookup_result = tf.nn.embedding_lookup(emb, index)
    return tf.reduce_mean(lookup_result, reduction_indices)

def embedding_lookup_sum(emb, index, reduction_indices=None, name=None):
  with tf.name_scope(name, 'emb_lookup_sum', [emb, index]):
    lookup_result = tf.nn.embedding_lookup(emb, index)
    return tf.reduce_sum(lookup_result, reduction_indices)

def masked_embedding_lookup(emb, index, reduction_indices=None, combiner='mean', exclude_zero_index=True, name=None):
  if combiner == 'mean':
    return masked_embedding_lookup_mean(emb, index, reduction_indices, combiner, exclude_zero_index, name)
  elif combiner == 'sum':
    return masked_embedding_lookup_sum(emb, index, reduction_indices, combiner, exclude_zero_index, name)
  else:
    raise ValueError('Unsupported combiner: ', combiner)

def masked_embedding_lookup_mean(emb, index, reduction_indices=None, exclude_zero_index=True, name=None):
  """
  @TODO need c++ op to really mask last dim zero feature vector
  now assume vector should zero filtered to be zero vector @TODO
  """
  with tf.name_scope(name, 'masked_emb_lookup_mean', [emb, index]):
    lookup_result = tf.nn.embedding_lookup(emb, index)
    if exclude_zero_index:
      masked_emb = mask2d(emb)
      mask_lookup_result = tf.nn.embedding_lookup(masked_emb, index)
      lookup_result = tf.mul(lookup_result, mask_lookup_result)
    return reduce_mean_with_mask(lookup_result,  
                                 tf.expand_dims(tf.cast(tf.sign(index), dtype=tf.float32), -1),
                                 reduction_indices)

def masked_embedding_lookup_sum(emb, index, reduction_indices=None, exclude_zero_index=True, name=None):
  """ 
  @TODO need c++ op to really mask last dim zero feature vector
  now assume vector should zero filtered  to be zero vector
  or to just make emb firt row zero before lookup ?
  """
  with tf.name_scope(name, 'masked_emb_lookup_sum', [emb, index]):
    lookup_result = tf.nn.embedding_lookup(emb, index)
    if exclude_zero_index:
      masked_emb = mask2d(emb)
      mask_lookup_result = tf.nn.embedding_lookup(masked_emb, index)
      lookup_result = tf.mul(lookup_result, mask_lookup_result)
    return tf.reduce_sum(lookup_result, reduction_indices)

def wrapped_embedding_lookup(emb, index, reduction_indices=None, combiner='mean', use_mask=False, name=None):
  """
  compare to embedding_lookup
  wrapped_embedding_lookup add use_mask
  """
  if use_mask:
    return masked_embedding_lookup(emb, index, reduction_indices, combiner, name)
  else:
    return embedding_lookup(emb, index, reduction_indices, combiner, name)

def batch_embedding_lookup_reduce(emb, index, combiner='mean', name=None):
  """
  same as embedding_lookup but use index_dim_length - 1 as reduction_indices
  """
  with tf.name_scope(name, 'batch_emb_lookup_%s'%combiner, [emb, index]):
    lookup_result = tf.nn.embedding_lookup(emb, index)
    #@NOTICE for tf.nn.embedding_lookup, index can be list.. here only tensor
    reduction_indices = len(index.get_shape()) - 1
    if combiner == 'mean':
      return tf.reduce_mean(lookup_result, reduction_indices)
    elif combiner == 'sum':
      return tf.reduce_sum(lookup_result, reduction_indices)
    else:
      raise ValueError('Unsupported combiner: ', combiner)

def batch_embedding_lookup_mean(emb, index, name=None):
  with tf.name_scope(name, 'batch_emb_lookup_mean', [emb, index]):
    lookup_result = tf.nn.embedding_lookup(emb, index)
    reduction_indices = len(index.get_shape()) - 1
    return tf.reduce_mean(lookup_result, reduction_indices)

def batch_embedding_lookup_sum(emb, index, name=None):
  with tf.name_scope(name, 'batch_emb_lookup_sum', [emb, index]):
    lookup_result = tf.nn.embedding_lookup(emb, index)
    reduction_indices = len(index.get_shape()) - 1
    return tf.reduce_sum(lookup_result, reduction_indices)

def batch_masked_embedding_lookup_reduce(emb, index, combiner='mean', exclude_zero_index=True, name=None):
  if combiner == 'mean':
    return batch_masked_embedding_lookup_mean(emb, index, exclude_zero_index, name)[1]
  elif combiner == 'sum':
    return batch_masked_embedding_lookup_sum(emb, index, exclude_zero_index, name)[1]
  else:
    raise ValueError('Unsupported combiner: ', combiner)

def batch_masked_embedding_lookup_and_reduce(emb, index, combiner='mean', exclude_zero_index=True, name=None):
  if combiner == 'mean':
    return batch_masked_embedding_lookup_mean(emb, index, exclude_zero_index, name)
  elif combiner == 'sum':
    return batch_masked_embedding_lookup_sum(emb, index, exclude_zero_index, name)
  else:
    raise ValueError('Unsupported combiner: ', combiner)

def batch_masked_embedding_lookup_mean(emb, index, exclude_zero_index=True, name=None):
  """
  @TODO need c++ op to really mask last dim zero feature vector
  now assume vector should zero filtered to be zero vector if not exclude_zero_index
  or will have to do lookup twice
  """
  with tf.name_scope(name, 'batch_masked_emb_lookup_mean', [emb, index]):
    #if exclude_zero_index:
    #-----so slow..
    #  emb = tf.concat(0, [tf.zeros([1, emb.get_shape()[1]]), 
    #                      emb[1:, :]])
    lookup_result = tf.nn.embedding_lookup(emb, index)
    reduction_indices = len(index.get_shape()) - 1
    if exclude_zero_index:
      #@TODO this will casue 4 times slower 
      masked_emb = mask2d(emb)    
      mask_lookup_result = tf.nn.embedding_lookup(masked_emb, index)
      lookup_result = tf.multiply(lookup_result, mask_lookup_result)
    return lookup_result, reduce_mean_with_mask(lookup_result,  
                                 tf.expand_dims(tf.cast(tf.sign(index), dtype=tf.float32), -1),
                                 reduction_indices)

def batch_masked_embedding_lookup_sum(emb, index, exclude_zero_index=True, name=None):
  """ 
  @TODO need c++ op to really mask last dim zero feature vector
  now assume vector should zero filtered to be zero vector if not exclude_zero_index
  or will have to do lookup twice
  """
  with tf.name_scope(name, 'batch_masked_emb_lookup_sum', [emb, index]):
    lookup_result = tf.nn.embedding_lookup(emb, index)
    reduction_indices = len(index.get_shape()) - 1  
    if exclude_zero_index:
      masked_emb = mask2d(emb)
      mask_lookup_result = tf.nn.embedding_lookup(masked_emb, index)
      lookup_result = tf.multiply(lookup_result, mask_lookup_result)
    return lookup_result, tf.reduce_sum(lookup_result, reduction_indices)

def batch_masked_embedding_lookup(emb, index, exclude_zero_index=True, name=None):
  """ 
  @TODO need c++ op to really mask last dim zero feature vector
  now assume vector should zero filtered to be zero vector if not exclude_zero_index
  or will have to do lookup twice
  """
  with tf.name_scope(name, 'batch_masked_emb_lookup_sum', [emb, index]):
    lookup_result = tf.nn.embedding_lookup(emb, index)
    reduction_indices = len(index.get_shape()) - 1  
    if exclude_zero_index:
      masked_emb = mask2d(emb)
      mask_lookup_result = tf.nn.embedding_lookup(masked_emb, index)
      lookup_result = tf.multiply(lookup_result, mask_lookup_result)
    return lookup_result 

def batch_wrapped_embedding_lookup(emb, index, combiner='mean', use_mask=False, exclude_zero_index=True, name=None):
  if use_mask:
    return batch_masked_embedding_lookup(emb, index, combiner, exclude_zero_index, name)
  else:
    return batch_embedding_lookup(emb, index, combiner, name)

def mask2d(emb):
  return tf.concat([tf.zeros([1, 1]), tf.ones([emb.get_shape()[0] - 1, 1])], 0)   

#-------x must >= 0  TODO  tf.not_equal(x, 0) ?
def length(x, dim=1):
  #return tf.reduce_sum(tf.sign(x), dim)
  return tf.reduce_sum(tf.to_int32(tf.sign(x)), dim)

def length2(x, dim=1):
  #len_ = tf.reduce_sum(tf.sign(x), dim)
  len_ = tf.reduce_sum(tf.to_int32(tf.sign(x)), dim)
  max_len = tf.cast(tf.reduce_max(len_), dtype=tf.int32)
  return len_, max_len

def full_length(x):
   return tf.ones([get_batch_size(x),], dtype=tf.int32) * tf.shape(x)[1]

#---------now only consider 2d @TODO
def dynamic_append(x, value=1):
  length = tf.reduce_sum(tf.sign(x), 1)
  rows = tf.range(tf.shape(x)[0])
  rows = tf.cast(rows, x.dtype)
  coords = tf.transpose(tf.stack([rows, length]))
  shape = tf.cast(tf.shape(x), x.dtype)
  delta = tf.sparse_to_dense(coords, shape, value)
  return x + delta

def dynamic_append_with_mask(x, mask, value=1):
  length = tf.reduce_sum(mask, 1)
  rows = tf.range(tf.shape(x)[0])
  rows = tf.cast(rows, x.dtype)
  coords = tf.transpose(tf.stack([rows, length]))
  shape = tf.cast(tf.shape(x), x.dtype)
  delta = tf.sparse_to_dense(coords, shape, value)
  return x + delta

def dynamic_append_with_length(x, length, value=1):
  rows = tf.range(tf.shape(x)[0])
  #rows = tf.cast(rows, x.dtype)
  rows = tf.cast(rows, length.dtype)
  coords = tf.transpose(tf.stack([rows, length]))
  #shape = tf.cast(tf.shape(x), x.dtype)
  shape = tf.cast(tf.shape(x), length.dtype)
  delta = tf.sparse_to_dense(coords, shape, value)
  return x + delta

def dynamic_append_with_length_float32(x, length, value=1):
  rows = tf.range(tf.shape(x)[0])
  rows = tf.cast(rows, length.dtype)
  coords = tf.transpose(tf.stack([rows, length]))
  shape = tf.cast(tf.shape(x), length.dtype)
  delta = tf.sparse_to_dense(coords, shape, 1)
  delta = tf.cast(delta, x.dtype) * value
  return x + delta

#@TODO not tested
def static_append(x, value=1):
  length = tf.reduce_sum(tf.sign(x), 1)
  rows = tf.range(x.get_shape()[0])
  rows = tf.cast(rows, x.dtype)
  coords = tf.transpose(tf.stack([rows, length]))
  shape = tf.cast(x.get_shape(), x.dtype)
  delta = tf.sparse_to_dense(coords, shape, value)
  return x + delta

def static_append_with_mask(x, mask, value=1):
  length = tf.reduce_sum(mask, 1)
  rows = tf.range(x.get_shape()[0])
  rows = tf.cast(rows, x.dtype)
  coords = tf.transpose(tf.stack([rows, length]))
  shape = tf.cast(x.get_shape(), x.dtype)
  delta = tf.sparse_to_dense(coords, shape, value)
  return x + delta

def static_append_with_length(x, length, value=1):
  rows = tf.range(x.get_shape()[0])
  rows = tf.cast(rows, x.dtype)
  coords = tf.transpose(tf.stack([rows, length]))
  shape = tf.cast(x.get_shape(), x.dtype)
  delta = tf.sparse_to_dense(coords, shape, value)
  return x + delta

def first_nrows(x, n):
  #eval_scores[0:num_evaluate_examples, 1] the diff is below you do not need other dim infos
  return tf.gather(x, range(n))

def exclude_last_col(x):
  """
  @TODO just hack since dynamic x[:,:-1] is not supported 
  now just work for 2d
  ref to https://github.com/tensorflow/tensorflow/issues/206
  """
  try:
    return x[:,:-1]
  except Exception:
    return tf.transpose(tf.gather(tf.transpose(x), tf.range(0, x.get_shape()[1] - 1)))

def dynamic_exclude_last_col(x):
  """
  @TODO just hack since dynamic x[:,:-1] is not supported 
  now just work for 2d
  ref to https://github.com/tensorflow/tensorflow/issues/206
  """
  try:
    #tf0.12 is ok here
    return x[:,:-1]
  except Exception:
    return tf.transpose(tf.gather(tf.transpose(x), tf.range(0, tf.shape(x)[1] - 1)))

def gather2d(x, idx):
  """
  from https://github.com/tensorflow/tensorflow/issues/206
  x = tf.constant([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])
idx = tf.constant([1, 0, 2])
idx_flattened = tf.range(0, x.shape[0]) * x.shape[1] + idx
y = tf.gather(tf.reshape(x, [-1]),  # flatten input
              idx_flattened)  # use flattened indices

with tf.Session(''):
  print y.eval()  # [2 4 9]
  """
  #FIXME
  try:
    idx_flattened = tf.cast(tf.range(0, x.shape[0]) * x.shape[1], idx.dtype) + idx
  except Exception:
    shape_ = tf.shape(x)
    idx_flattened = tf.cast(tf.range(0, shape_[0]) * shape_[1], idx.dtype) + idx
  y = tf.gather(tf.reshape(x, [-1]),  # flatten input
                idx_flattened)  # use flattened indices
  return y

#deprecated gpu mem issue, use tf.gather_nd instead
def dynamic_gather2d(x, idx):
  #FIMXE
  idx_flattened = tf.cast(tf.range(0, tf.shape(x)[0]) * tf.shape(x)[1], idx.dtype) + idx
  y = tf.gather(tf.reshape(x, [-1]),  # flatten input
                idx_flattened)  # 
  return y

def subtract_by_diff(x, y):
  """
  [1,2, 3, 4] - [1, 2, 1, 4] = [1, 2, 2, 4]
  assume input x, y is not the same
  @TODO c++ op
  """
  delta = tf.abs(x - y)
  delta_bool = tf.cast(delta, tf.bool)
  return tf.add(tf.mul(x, tf.cast(tf.logical_not(delta_bool), x.dtype)), delta)

#------this can only deal with same first dimension.. if two batch different batch size not ok
# def _align(x, y, dim):
#   x_shape = tf.shape(x)
#   y_shape = tf.shape(y)
#   padding_shape = subtract_by_diff(x_shape, y_shape)
#   x, y = tf.cond(
#     tf.greater(x_shape[dim], y_shape[dim]), 
#     lambda: (x, tf.concat(dim, [y, tf.zeros(padding_shape, x.dtype)])), 
#     lambda: (tf.concat(dim, [x, tf.zeros(padding_shape, x.dtype)]), y))
#   return x, y

# def align(x, y, dim):
#   """
#   @TODO use c++ op
#   """
#   x_shape = tf.shape(x)
#   y_shape = tf.shape(y)
#   x, y = tf.cond(
#     tf.equal(x_shape[dim], y_shape[dim]),
#     lambda: (x, y),
#     lambda: _align(x, y, dim))
#   return x, y


def _align_col_padding2d(x, y):
  x_shape = tf.shape(x)
  y_shape = tf.shape(y)

  x, y = tf.cond(
    tf.greater(x_shape[1], y_shape[1]), 
    lambda: (x, tf.pad(y, [[0, 0], [0, x_shape[1] - y_shape[1]]])), 
    lambda: (tf.pad(x, [[0, 0], [0, y_shape[1] - x_shape[1]]]), y))
  return x, y

def align_col_padding2d(x, y):
  x_shape = tf.shape(x)
  y_shape = tf.shape(y)
  x, y = tf.cond(
    tf.equal(x_shape[1], y_shape[1]),
    lambda: (x, y),
    lambda: _align_col_padding2d(x, y))
  return x, y

def make_batch_compat(sequence):
  sequence_length = length(sequence)
  num_steps = tf.to_int32(tf.reduce_max(sequence_length))
  sequence = sequence[:, :num_steps]
  return sequence

def last_relevant(output, length):
  """
  https://danijar.com/variable-sequence-lengths-in-tensorflow/
  Select the Last Relevant Output
  For sequence classification, we want to feed the last output of the recurrent network into a predictor, e.g. a softmax layer. While taking the last frame worked well for fixed-sized sequences, we not have to select the last relevant frame. This is a bit cumbersome in TensorFlow since it does’t support advanced slicing yet. In Numpy this would just be output[:, length - 1]. But we need the indexing to be part of the compute graph in order to train the whole system end-to-end.
  @TODO understand below code
  """
  batch_size = tf.shape(output)[0]
  #@TODO could not use in rnn.py why even int fixed length mode?  max_length = int(output.get_shape()[1]) __int__ returned non-int (type NoneType) 
  #because even though you convert sparse to dense to make same length, that length is dynamic , if get in static will be None? So if you use FixedLen to read tfrecord like version 0 in 
  #models/image-text-sim then might ok here, but for comment seems using feed_dict mode still None why? placeholder should have fixed length! 
  #@TODO why need int otherwise in tf.reshape(output, [-1, out_size]) TypeError: Expected int32, got Dimension(1024) of type 'Dimension' instead.
  max_length = int(output.get_shape()[1])
  out_size = int(output.get_shape()[2])
  #index = tf.range(0, batch_size) * max_length + (length - 1)
  #@TODO may be it is best to convert tfrecord reading to int64 and convert to int32 to avoid unnecessary cast
  index = tf.cast(tf.range(0, batch_size), length.dtype) * max_length + (length - 1)
  flat = tf.reshape(output, [-1, out_size])
  relevant = tf.gather(flat, index)
  return relevant


#----------for rnn
def dynamic_last_relevant(output, length):
  """
  https://danijar.com/variable-sequence-lengths-in-tensorflow/
  Select the Last Relevant Output
  For sequence classification, we want to feed the last output of the recurrent network into a predictor, e.g. a softmax layer. While taking the last frame worked well for fixed-sized sequences, we not have to select the last relevant frame. This is a bit cumbersome in TensorFlow since it does’t support advanced slicing yet. In Numpy this would just be output[:, length - 1]. But we need the indexing to be part of the compute graph in order to train the whole system end-to-end. 

  not this will only work for 3 d, for general pupose dynamic last might consider
  output = tf.reverse_sequence(output, seqence_lenth, 1)
  return output[:, 0, :]
  """
  shape = tf.shape(output)
  batch_size = shape[0]
  #max_length = shape[1]
  max_length = tf.cast(shape[1], length.dtype)
  #out_size = shape[2]
  out_size = int(output.get_shape()[2])
  #index = tf.range(0, batch_size) * max_length + (length - 1)
  #@TODO may be it is best to convert tfrecord reading to int64 and convert to int32 to avoid unnecessary cast
  #here length might be tf.int64 if length calced from mask of int64 type
  index = tf.cast(tf.range(0, batch_size), length.dtype) * max_length + (length - 1)
  flat = tf.reshape(output, [-1, out_size])
  relevant = tf.gather(flat, index)
  return relevant

def dynamic_last(output):
  """
  https://danijar.com/variable-sequence-lengths-in-tensorflow/
  Select the Last Relevant Output
  For sequence classification, we want to feed the last output of the recurrent network into a predictor, e.g. a softmax layer. While taking the last frame worked well for fixed-sized sequences, we not have to select the last relevant frame. This is a bit cumbersome in TensorFlow since it does’t support advanced slicing yet. In Numpy this would just be output[:, length - 1]. But we need the indexing to be part of the compute graph in order to train the whole system end-to-end.
  """
  shape = tf.shape(output)
  batch_size = shape[0]
  max_length = shape[1]
  #out_size = shape[2]
  out_size = int(output.get_shape()[2])
  #index = tf.range(0, batch_size) * max_length + (length - 1)
  #@TODO may be it is best to convert tfrecord reading to int64 and convert to int32 to avoid unnecessary cast
  index = tf.range(0, batch_size) * max_length + (max_length - 1)
  flat = tf.reshape(output, [-1, out_size])
  relevant = tf.gather(flat, index)
  return relevant

def static_last(output):
  return output[:, int(output.get_shape()[1]) - 1, :]

#http://stackoverflow.com/questions/37670886/gathering-columns-of-a-2d-tensor-in-tensorflow
def gather_cols(params, indices, name=None):
    """Gather columns of a 2D tensor.

    Args:
        params: A 2D tensor.
        indices: A 1D tensor. Must be one of the following types: ``int32``, ``int64``.
        name: A name for the operation (optional).

    Returns:
        A 2D Tensor. Has the same type as ``params``.
    """
    with tf.name_scope(name, "gather_cols", [params, indices]) as scope:
        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = tf.convert_to_tensor(indices, name="indices")
        #indices = tf.to_int32(indices)
        try:
            params.get_shape().assert_has_rank(2)
        except ValueError:
            raise ValueError('\'params\' must be 2D.')
        try:
            indices.get_shape().assert_has_rank(1)
        except ValueError:
            raise ValueError('\'params\' must be 1D.')

        # Define op
        #int64 will cause Tensor conversion requested dtype int64 for Tensor with dtype int32
        indices = tf.to_int32(indices) 

        p_shape = tf.shape(params)
        p_flat = tf.reshape(params, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],
                                       [-1, 1]) + indices, [-1])
        return tf.reshape(tf.gather(p_flat, i_flat),
                          [p_shape[0], -1])

#[batch_size, num_steps, emb_dim] * [emb_dim, vocab_size] -> [batch_size, num_steps, vocab_size] if keep_dims
#else [batch_size * num_steps, vocab_size]
def batch_matmul_embedding(x, emb, keep_dims=False):
  batch_size = tf.shape(x)[0]
  emb_shape = tf.shape(emb)
  emb_dim = emb_shape[0]
  vocab_size = emb_shape[1]
  x = tf.reshape(x, [-1, emb_dim])
  logits = tf.matmul(x, emb)
  if keep_dims:
    logits = tf.reshape(logits, [batch_size, -1, vocab_size])
  return logits


def constants(value, shape, dtype=dtypes.float32, name=None):
  """Creates a tensor with all elements set to zero.

  This operation returns a tensor of type `dtype` with shape `shape` and
  all elements set to zero.

  For example:

  ```python
  tf.zeros([3, 4], tf.int32) ==> [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
  ```

  Args:
    shape: Either a list of integers, or a 1-D `Tensor` of type `int32`.
    dtype: The type of an element in the resulting `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with all elements set to zero.
  """
  dtype = dtypes.as_dtype(dtype).base_dtype
  with ops.name_scope(name, "constants", [shape]) as name:
    try:
      shape = tensor_shape.as_shape(shape)
      output = constant(value, shape=shape, dtype=dtype, name=name)
    except (TypeError, ValueError):
      shape = ops.convert_to_tensor(shape, dtype=dtypes.int32, name="shape")
      output = fill(shape, constant(value, dtype=dtype), name=name)
  assert output.dtype.base_dtype == dtype
  return output


def constants_like(tensor, value, dtype=None, name=None, optimize=True):
  """Creates a tensor with all elements set to zero.

  Given a single tensor (`tensor`), this operation returns a tensor of the
  same type and shape as `tensor` with all elements set to zero. Optionally,
  you can use `dtype` to specify a new type for the returned tensor.

  For example:

  ```python
  # 'tensor' is [[1, 2, 3], [4, 5, 6]]
  tf.zeros_like(tensor) ==> [[0, 0, 0], [0, 0, 0]]
  ```

  Args:
    tensor: A `Tensor`.
    dtype: A type for the returned `Tensor`. Must be `float32`, `float64`,
    `int8`, `int16`, `int32`, `int64`, `uint8`, `complex64`, or `complex128`.
    name: A name for the operation (optional).
    optimize: if true, attempt to statically determine the shape of 'tensor'
    and encode it as a constant.

  Returns:
    A `Tensor` with all elements set to zero.
  """
  with ops.name_scope(name, "constants_like", [tensor]) as name:
    tensor = ops.convert_to_tensor(tensor, name="tensor")
    if dtype is not None and tensor.dtype != dtype:
      ret = constants(value, shape_internal(tensor, optimize=optimize), dtype, name=name)
      ret.set_shape(tensor.get_shape())
      return ret
    else:
      #TODO better handle
      return gen_array_ops._zeros_like(tensor, name=name) + value


#-----------loss ops
#@TODO contrib\losses\python\losses\loss_ops.py
def sparse_softmax_cross_entropy(x, y):
  return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=y))

def softmax_cross_entropy(x, y):
  return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y))

def sigmoid_cross_entropy(x, y):
  return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y))

activations = {'sigmoid' :  tf.nn.sigmoid, 'tanh' : tf.nn.tanh, 'relu' : tf.nn.relu}


#--@TODO other rank loss
#Depreciated use melt.losses.

def reduce_loss(loss_matrix, combiner='mean'):
  if combiner == 'mean':
    return tf.reduce_mean(loss_matrix)
  else:
    return tf.reduce_sum(loss_matrix)

def hinge_loss(pos_score, neg_score, margin=0.1, combiner='mean', name=None):
  with tf.name_scope(name, 'hinge_loss', [pos_score, neg_score]):
    loss_matrix = tf.maximum(0., margin - (pos_score - neg_score))
    loss = reduce_loss(loss_matrix, combiner)
    return loss

def cross_entropy_loss(scores, num_negs=1, combiner='mean', name=None):
  with tf.name_scope(name, 'cross_entropy_loss', [scores]):
    batch_size = scores.get_shape()[0]
    targets = tf.concat([tf.ones([batch_size, 1], tf.float32), tf.zeros([batch_size, num_negs], tf.float32)], 1)
    #http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/ 
    #I think for binary is same for sigmoid or softmax
    logits = tf.sigmoid(scores)
    loss_matrix = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets)
    #loss_matrix = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    loss = reduce_loss(loss_matrix, combiner)
    return loss

def hinge_cross_loss(pos_score, neg_score, combiner='mean', name=None):
  with tf.name_scope(name, 'hinge_cross_loss', [pos_score, neg_score]):
    logits = pos_score - neg_score
    logits = tf.sigmoid(logits)
    targets = tf.ones_like(neg_score, tf.float32)
    loss_matrix = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets)
    loss = reduce_loss(loss_matrix, combiner)
    return loss

def last_dimension(x):
  return tf_utils.last_dimension(x.get_shape())

def first_dimension(x):
  return tf_utils.first_dimension(x.get_shape())

def dimension(x, index):
  return x.get_shape()[index].value

#def batch_values_to_indices(x):
#  shape_ = tf.shape(x)
#  batch_size = shape_[0]
#  num_cols = shape_[1]
#  d = tf.transpose(tf.expand_dims(tf.range(batch_size),0))
#  d = tf.tile(d, [1, num_cols])
#  d_flatten = tf.reshape(d, [-1, 1])
#  x_flatten = tf.reshape(x, [-1, 1])
#  r_flatten = tf.concat([d_flatten, x_flatten], 1)
#  return tf.reshape(r_flatten, [shape_[0], shape_[1], 2])

#[[0,1,2], [3,4,5]] - > [[[0,0], [0,1], [0,2]], [[1,3], [1,4], [1,5]]]
def batch_values_to_indices(index_matrix):
  replicated_first_indices = tf.range(tf.shape(index_matrix)[0])
  rank = len(index_matrix.get_shape())
  if rank == 2:
    replicated_first_indices = tf.tile(
        tf.expand_dims(replicated_first_indices, dim=1),
        [1, tf.shape(index_matrix)[1]])
  return tf.stack([replicated_first_indices, index_matrix], axis=rank)

def to_nd_indices(index_matrix):
  replicated_first_indices = tf.range(tf.shape(index_matrix)[0])
  rank = len(index_matrix.get_shape())
  if rank == 2:
    replicated_first_indices = tf.tile(
        tf.expand_dims(replicated_first_indices, dim=1),
        [1, tf.shape(index_matrix)[1]])
  #print('--------', replicated_first_indices, index_matrix)
  return tf.stack([replicated_first_indices, index_matrix], axis=rank)

def nhot(x, max_dim):
  return tf.scatter_nd(to_nd_indices(tf.to_int32(x)), tf.ones_like(x), shape=(get_batch_size(x), max_dim))

def dense(inputs, kernel, bias=None, activation=None):
  #inputs = ops.convert_to_tensor(inputs, dtype=dtype)
  shape = inputs.get_shape().as_list()
  output_shape = shape[:-1] + [kernel.get_shape().as_list()[-1]]
  if len(output_shape) > 2:
    # Broadcasting is required for the inputs.
    outputs = standard_ops.tensordot(inputs, kernel, [[len(shape) - 1],
                                                           [0]])
    # Reshape the output back to the original ndim of the input.
    outputs.set_shape(output_shape)
  else:
    outputs = standard_ops.matmul(inputs, kernel)
  if bias is not None:
    outputs = nn.bias_add(outputs, bias)
  if activation is not None:
    return activation(outputs)  # pylint: disable=not-callable
  return outputs

def sequence_equal(x, y):
  return tf.reduce_mean(tf.to_int32(tf.equal(x,y)), 1)

def get_batch_size(x):
  #or .shape.as_list()[0]  or .get_shape().as_list()[0]
  return x.get_shape()[0].value or tf.shape(x)[0]

def get_shape(x, dim):
  return x.get_shape()[dim].value or tf.shape(x)[dim]

def get_dims(x):
  return len(x.get_shape())

def get_weighted_outputs(outputs, sequence_length):
  weight = -1e18
  sequence_mask = tf.expand_dims(1. - tf.to_float(tf.sequence_mask(sequence_length, tf.shape(outputs)[1])), -1)
  weighted_mask = sequence_mask * weight
  weighted_outputs = outputs + weighted_mask  
  return weighted_outputs

# default axis might be -2
def max_pooling(outputs, sequence_length=None, axis=1, reduce_func=tf.reduce_max):
  if sequence_length is None:
    return reduce_func(outputs, axis)
  weight = -1e18
  sequence_mask = tf.expand_dims(1. - tf.to_float(tf.sequence_mask(sequence_length, tf.shape(outputs)[1])), -1)
  weighted_mask = sequence_mask * weight
  weighted_output = outputs + weighted_mask
  return reduce_func(weighted_output, axis)

def max_pooling2(outputs, sequence_length, sequence_length2, axis=1, reduce_func=tf.reduce_max):
  weight = -1e18
  sequence_mask = tf.expand_dims(1. - tf.to_float(tf.sequence_mask(sequence_length)), -1)
  sequence_mask2 = tf.expand_dims(1. - tf.to_float(tf.sequence_mask(sequence_length2)), -1)

  sequence_mask = tf.concat([sequence_mask, sequence_mask2], 1)
  weighted_mask = sequence_mask * weight
  weighted_output = outputs + weighted_mask
  return reduce_func(weighted_output, axis)

# not to use it directly!
def top_k_pooling(outputs, top_k, sequence_length=None, axis=1):
  if sequence_length is None:
    return reduce_func(outputs, axis)
  weight = -1e18
  sequence_mask = tf.expand_dims(1. - tf.to_float(tf.sequence_mask(sequence_length, tf.shape(outputs)[1])), -1)
  weighted_mask = sequence_mask * weight
  weighted_output = outputs + weighted_mask
  # swap last two dimensions since top_k will be applied along the last dimension
  shifted_output = tf.transpose(weighted_output, [0, 2, 1])
  return tf.nn.top_k(shifted_output, top_k)

import melt
def argtopk_pooling(outputs, top_k, sequence_length=None, axis=1):
  x = top_k_pooling(outputs, top_k, sequence_length, axis).indices
  #return tf.reshape(x, [melt.get_shape(outputs, 0), -1])
  return tf.reshape(x, [-1, melt.get_shape(outputs, -1) * top_k])

def argmax_pooling(outputs, sequence_length, axis=1):
  return max_pooling(outputs, sequence_length, axis, reduce_func=tf.argmax)

def mean_pooling(outputs, sequence_length=None, axis=1):
  if sequence_length is None:
    return tf.reduce_mean(outputs, axis)
  sequence_mask = tf.to_float(tf.expand_dims(tf.sequence_mask(sequence_length), -1))
  outputs = outputs * sequence_mask
  return tf.reduce_sum(outputs, axis) / tf.to_float(tf.expand_dims(sequence_length, 1)) 

def sum_pooling(outputs, sequence_length=None, axis=1):
  if sequence_length is None:
    return tf.reduce_sum(outputs, axis)
  sequence_mask = tf.to_float(tf.expand_dims(tf.sequence_mask(sequence_length), -1))
  outputs = outputs * sequence_mask
  return tf.reduce_sum(outputs, axis) 

def hier_encode(outputs, sequence_length, window_size=3, axis=1):
  #weight = -1e18
  #sequence_mask = tf.expand_dims(1. - tf.to_float(tf.sequence_mask(sequence_length, tf.shape(outputs)[1])), -1)
  #weighted_mask = sequence_mask * weight
  #print('before mask', outputs)
  #outputs = outputs + weighted_mask
  #print('before_pool', outputs)
  sequence_mask = tf.to_float(tf.expand_dims(tf.sequence_mask(sequence_length), -1))
  outputs = outputs * sequence_mask
  #outputs = get_weighted_outputs(outputs, sequence_length)
  outputs = tf.nn.pool(outputs, [window_size], 'AVG', 'SAME')
  return outputs


def hier_pooling(outputs, sequence_length, window_size=3, axis=1):
  #weight = -1e18
  #sequence_mask = tf.expand_dims(1. - tf.to_float(tf.sequence_mask(sequence_length, tf.shape(outputs)[1])), -1)
  #weighted_mask = sequence_mask * weight
  #print('before mask', outputs)
  #outputs = outputs + weighted_mask
  #print('before_pool', outputs)
  sequence_mask = tf.to_float(tf.expand_dims(tf.sequence_mask(sequence_length), -1))
  outputs = outputs * sequence_mask
  #outputs = get_weighted_outputs(outputs, sequence_length)
  outputs = tf.nn.pool(outputs, [window_size], 'AVG', 'SAME')
  outputs = get_weighted_outpus(outputs, sequence_length)
  #print('------------outputs', outputs)
  outputs = tf.reduce_max(outputs, axis)
  return outputs

def argmax_importance(argmax_values, shape):
  # argmax_vlalues [batch_size, emb_dim]
  indices = batch_values_to_indices(tf.to_int32(argmax_values))
  updates = tf.ones_like(argmax_values)
  # actually do not need sequence mask ? since when encode has masked already
  # TODO check
  scores = tf.scatter_nd(indices, updates, shape=shape) 
  #scores = tf.scatter_nd(indices, updates, shape=shape) * tf.to_int64(tf.sequence_mask(self.sequence_length, shape[-1]))
  return scores

def maxpooling_importance(outputs, sequence_length=None, axis=1):
  argmax_values = argmax_pooling(outputs, sequence_length, axis)
  return argmax_importance(argmax_values, [tf.shape(outputs)[0], tf.shape(outputs)[1]])

def topkpooling_importance(outputs, top_k, sequence_length=None, axis=1):
  argtopk_values = argtopk_pooling(outputs, top_k, sequence_length, axis)
  return argmax_importance(argtopk_values, [tf.shape(outputs)[0], tf.shape(outputs)[1] * top_k])


# TODO like tf change from dim to axis
def slim_batch(sequence, sequence_length=None, dim=1):
  if sequence_length is None:
    sequence_length = length(sequence, dim)
  num_steps = tf.cast(tf.reduce_max(sequence_length), dtype=tf.int32)
  dims = get_dims(sequence)
  starts = [0] * dims 
  lens = [-1] * dims 
  lens[dim] = num_steps
  sequence = tf.slice(sequence, starts, lens) 
  return sequence  

def slim_batch2(sequence, sequence_length=None, dim=1):
  if sequence_length is None:
    sequence_length = length(sequence, dim)
  num_steps = tf.cast(tf.reduce_max(sequence_length), dtype=tf.int32)
  dims = get_dims(sequence)
  starts = [0] * dims 
  lens = [-1] * dims 
  lens[dim] = num_steps
  sequence = tf.slice(sequence, starts, lens) 
  return sequence, sequence_length  

# from squad of HKUST
def dropout(args, keep_prob, training, mode="recurrent"):
  if keep_prob < 1.0 and training:
    noise_shape = None
    scale = 1.0
    shape = tf.shape(args)
    if mode == "embedding":
      noise_shape = [shape[0], 1]
      scale = keep_prob
    if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
      noise_shape = [shape[0], 1, shape[-1]]    
    args = tf.nn.dropout(args, keep_prob, noise_shape=noise_shape) * scale
  return args

# this is fine, but lengths should has one row without padding 0
def masked_softmax(values, lengths):
  with tf.variable_scope('masked_softmax'):
    mask = tf.expand_dims(tf.sequence_mask(lengths, dtype=tf.float32), -1)
    # here mask 1 will get nan mask 0 -inf
    inf_mask = (1 - mask) * -np.inf
    # keep -inf unchanged and make nan to 0
    inf_mask = tf.where(tf.is_nan(inf_mask), tf.zeros_like(inf_mask), inf_mask)
    return tf.nn.softmax(tf.multiply(values, mask) + inf_mask, axis=1)

# TODO 
def softmax_mask(val, mask):
  INF = 1e30
  return -INF * (1 - tf.cast(mask, tf.float32)) + val

def get_words_importance(outputs=None, sequence_length=None, top_k=None, method='max'):
  if method == 'max':
    words_scores = maxpooling_importance(outputs, sequence_length) 
    words_scores = tf.to_float(words_scores / tf.reduce_sum(words_scores, -1)) * tf.to_float(sequence_length)
  elif method == 'attention' or method =='att':
    words_scores = tf.get_collection('self_attention')[-1] * tf.to_float(sequence_length)
  elif method == 'top_k' or method == 'topk':
    words_scores = topkpooling_importance(outputs, top_k, sequence_length) 
    words_scores = tf.to_float(words_scores / tf.reduce_sum(words_scores, -1)) * tf.to_float(sequence_length)
  else:
    #raise ValueError(f'unsupported method {method}')
    words_scores = []
  return words_scores