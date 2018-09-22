#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   layers.py
#        \author   chenghuige  
#          \date   2016-08-19 23:22:44.032101
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import melt
#since not from melt.layers.layers import * this is safe
from tensorflow.contrib.layers.python.layers import utils
import tensorflow.contrib.slim as slim

import functools
import six

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.layers import convolutional as convolutional_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import  normalization as normalization_layers
from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import moving_averages

logging = melt.logging

def fully_connected(inputs,
                    num_outputs,
                    input_dim=None,
                    activation_fn=nn.relu,
                    normalizer_fn=None,
                    normalizer_params=None,
                    weights_initializer=initializers.xavier_initializer(),
                    weights_regularizer=None,
                    biases_initializer=init_ops.zeros_initializer(),
                    biases_regularizer=None,
                    reuse=None,
                    variables_collections=None,
                    outputs_collections=None,
                    trainable=True,
                    scope=None):

  use_bias = biases_initializer is not None

  #--------TODO: use commented code as layers.fully_connected then, for app code you must manully pass scope like 'mlp' otherwise evaluate will fail try to use Mlp_1 not resue=True b
  #--------by tf.variable_scope.reuse_variables() see http://stackoverflow.com/questions/40536665/tensorflow-varscope-reuse-variables
  #with variable_scope.variable_scope(
  #  scope, 'Mlp', [inputs],
  #  reuse=reuse) as vs:
  with tf.variable_scope(scope, 'fully_connected', [inputs], reuse=reuse):
    is_dense_input = True if isinstance(inputs, tf.Tensor) else False
    dtype=inputs.dtype.base_dtype if is_dense_input else inputs[1].values.dtype.base_dtype
    #sparse input must tell input_dim
    assert is_dense_input or input_dim is not None 
    if is_dense_input:
      shape = inputs.get_shape().as_list() 
      input_dim = shape[-1].value
      assert len(shape) == 2, "now only consider X shape dim as 2, TODO: make > 2 ok like layers.fully_connected"

    #-----------deal first hidden
    if is_dense_input:
     w_h =  tf.get_variable('weights',
                            shape=[input_dim, num_outputs],
                            initializer=weights_initializer,
                            regularizer=weights_regularizer,
                            dtype=dtype,
                            trainable=trainable)
    else:
     with tf.device('/cpu:0'):
       w_h =  tf.get_variable('weights',
                              shape=[input_dim, num_outputs],
                              initializer=weights_initializer,
                              regularizer=weights_regularizer,
                              dtype=dtype,
                              trainable=trainable)

    if use_bias:
     b_h = tf.get_variable('biases',
                           shape=[num_outputs,],
                           initializer=biases_initializer,
                           regularizer=biases_regularizer,
                           dtype=dtype,
                           trainable=trainable)

    outputs = melt.matmul(inputs, w_h)
    if use_bias:
     outputs = nn.bias_add(outputs, b_h)
    if activation_fn is not None:
     outputs = activation_fn(outputs)  # pylint: disable=not-callable

    return outputs

linear = functools.partial(fully_connected, activation_fn=None)

def mlp(x, hidden_size, output_size, activation=tf.nn.relu, scope=None):
  scope = 'mlp' if scope is None else scope
  with tf.variable_scope(scope):
    hidden = fully_connected(x, hidden_size, activation)
    w_o = melt.get_weights('w_o', [hidden_size, output_size])
    b_o = melt.get_bias('b_o', [output_size])
    return tf.nn.xw_plus_b(hidden, w_o, b_o)

def mlp_nobias(x, hidden_size, output_size, activation=tf.nn.relu, scope=None):
  scope = 'mlp_nobias' if scope is None else scope
  with tf.variable_scope(scope):
    input_dim = utils.last_dimension(x.get_shape(), min_rank=2)
    if isinstance(x, tf.Tensor):
      w_h = melt.get_weights('w_h', [input_dim, hidden_size])
    else:
      with tf.device('/cpu:0'):
        w_h = melt.get_weights('w_h', [input_dim, hidden_size]) 
    w_o = melt.get_weights('w_o', [hidden_size, output_size])
    return  melt.mlp_forward_nobias(x, w_h, w_o, activation)


#def self_attention(outputs, sequence_length, hidden_size, activation=tf.nn.tanh, scope='self_attention'):
def self_attention(outputs, sequence_length, hidden_size=128, activation=tf.nn.relu, scope='self_attention'):
  with tf.variable_scope('self_attention'):
    hidden_layer = tf.layers.dense(outputs, hidden_size, activation=activation)
    logits = tf.layers.dense(hidden_layer, 1, activation=None)
    alphas = melt.masked_softmax(logits, sequence_length)
    encoding = tf.reduce_sum(outputs * alphas, 1)
    # [batch_size, sequence_length, 1] -> [batch_size, sequence_length]
    alphas = tf.squeeze(alphas) 
    return encoding, alphas

def self_attention_outputs(outputs, sequence_length, hidden_size=128, activation=tf.nn.relu, scope='self_attention'):
  with tf.variable_scope('self_attention'):
    hidden_layer = tf.layers.dense(outputs, hidden_size, activation=activation)
    logits = tf.layers.dense(hidden_layer, 1, activation=None)
    alphas = melt.masked_softmax(logits, sequence_length)
    outputs = outputs * alphas
    return outputs

#def self_attention(outputs, sequence_length, hidden_size, activation=tf.nn.tanh, scope='self_attention'):
def attention_layer(outputs, sequence_length, hidden_size=128, activation=tf.nn.relu, scope='self_attention'):
  with tf.variable_scope('self_attention'):
    hidden_layer = tf.layers.dense(outputs, hidden_size, activation=activation)
    logits = tf.layers.dense(hidden_layer, 1, activation=None)
    alphas = melt.masked_softmax(logits, sequence_length)
    encoding = tf.reduce_sum(outputs * alphas, 1)
    # [batch_size, sequence_length, 1] -> [batch_size, sequence_length]
    #alphas = tf.squeeze(alphas) 
    return encoding, alphas

def attention_outputs(outputs, sequence_length, hidden_size=128, activation=tf.nn.relu, scope='self_attention'):
  with tf.variable_scope('self_attention'):
    hidden_layer = tf.layers.dense(outputs, hidden_size, activation=activation)
    logits = tf.layers.dense(hidden_layer, 1, activation=None)
    alphas = melt.masked_softmax(logits, sequence_length)
    outputs = outputs * alphas
    return outputs

def batch_norm(x, is_training=False, name=''):  
  return tf.contrib.layers.batch_norm(inputs=x,
                                      decay=0.95,
                                      center=True,
                                      scale=True,
                                      is_training=is_training,
                                      updates_collections=None,
                                      fused=True,
                                      scope=(name + 'batch_norm')) 

import melt 
from melt import dropout, softmax_mask

def gate_layer(res, keep_prob=1.0, is_train=None, scope='gate'):
  with tf.variable_scope(scope):
    dim = melt.get_shape(res, -1)
    d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
    gate = tf.layers.dense(d_res, dim, use_bias=False, activation=tf.nn.sigmoid, name=scope)
    return res * gate

def gate(res, keep_prob=1.0, is_train=None, scope='gate'):
  with tf.variable_scope(scope):
    dim = melt.get_shape(res, -1)
    d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
    gate = tf.layers.dense(d_res, dim, use_bias=False, activation=tf.nn.sigmoid, name=scope)
    return res * gate

# DO not only use semantic_fusion without droput, overfit prone
def semantic_fusion(input_vector, fusion_vectors, scope='semantic_fusion'):
  """Runs a semantic fusion unit on the input vector and the fusion vectors to produce an output. Input: input_vector: Vector of size [batch_size, ..., input_dim]. This vector must have the same size as each of the vectors in the fusion_vectors list. fusion_vectors: List of vectors of size [batch_size, ..., input_dim]. The vectors in this list must all have the same size as the input_vector. input_dim: The last dimension of the vectors (python scalar) Output: Vector of the same size as the input_vector. """
  with tf.variable_scope(scope):
      assert len(fusion_vectors) > 0
      vectors = tf.concat([input_vector] + fusion_vectors, axis=-1) # size = [batch_size, ..., input_dim * (len(fusion_vectors) + 1)]
      dim = melt.get_shape(input_vector, -1)
      r =  tf.layers.dense(vectors, dim, use_bias=True, activation=tf.nn.tanh, name='composition')
      g = tf.layers.dense(vectors, dim, use_bias=True, activation=tf.nn.sigmoid, name='gate')
      return g * r + (1 - g) * input_vector 

def semantic_fusion_combine(x, y):
  if melt.get_shape(x, -1) != melt.get_shape(y, -1):
    y = tf.layers.dense(y, melt.get_shape(x, -1), activation=None, name='semantic_fusion_fc')
  return semantic_fusion(x, [y, x * y, x - y])

def dsemantic_fusion(input_vector, fusion_vectors, keep_prob=1.0, is_train=None, scope='semantic_fusion'):
  """Runs a semantic fusion unit on the input vector and the fusion vectors to produce an output. Input: input_vector: Vector of size [batch_size, ..., input_dim]. This vector must have the same size as each of the vectors in the fusion_vectors list. fusion_vectors: List of vectors of size [batch_size, ..., input_dim]. The vectors in this list must all have the same size as the input_vector. input_dim: The last dimension of the vectors (python scalar) Output: Vector of the same size as the input_vector. """
  with tf.variable_scope(scope):
      assert len(fusion_vectors) > 0
      vectors = tf.concat([input_vector] + fusion_vectors, axis=-1) # size = [batch_size, ..., input_dim * (len(fusion_vectors) + 1)]
      dim = melt.get_shape(input_vector, -1)
      dv = dropout(vectors, keep_prob=keep_prob, is_train=is_train)
      r =  tf.layers.dense(dv, dim, use_bias=True, activation=tf.nn.tanh, name='composition')
      g = tf.layers.dense(dv, dim, use_bias=True, activation=tf.nn.sigmoid, name='gate')
      return g * r + (1 - g) * input_vector 

def dsemantic_fusion_combine(x, y, keep_prob=1.0, is_train=None):
  if melt.get_shape(x, -1) != melt.get_shape(y, -1):
    y = tf.layers.dense(y, melt.get_shape(x, -1), activation=None, name='semantic_fusion_fc')
  return dsemantic_fusion(x, [y, x * y, x - y], keep_prob, is_train)

def dsemantic_fusion_simple_combine(x, y, keep_prob=1.0, is_train=None):
  if melt.get_shape(x, -1) != melt.get_shape(y, -1):
    y = tf.layers.dense(y, melt.get_shape(x, -1), activation=None, name='semantic_fusion_simple_fc')
  return dsemantic_fusion(x, [y], keep_prob, is_train, scope='semantic_fusion_simple')  
  
# # used in toxic for self attetion, from squad HSTKU rnet code
# # TODO dot attention semantic fu
# def dot_attention(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, scope="dot_attention"):
#   with tf.variable_scope(scope):
#     d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
#     d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
#     JX = tf.shape(inputs)[1]
    
#     with tf.variable_scope("attention"):
#       inputs_ = tf.layers.dense(d_inputs, hidden, use_bias=False, activation=tf.nn.relu, name='inputs')
#       memory_ = tf.layers.dense(d_memory, hidden, use_bias=False, activation=tf.nn.relu, name='memory')
#       # [batch, input_len, meory_len]
#       outputs = tf.matmul(inputs_, tf.transpose(memory_, [0, 2, 1])) / (hidden ** 0.5)
#       if mask is not None:
#         mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
#         logits = tf.nn.softmax(softmax_mask(outputs, mask))
#       else:
#         logits = tf.nn.softmax(outputs)
#       outputs = tf.matmul(logits, memory)

#       res = tf.concat([inputs, outputs], axis=2)

#     # self attention also using gate ? TODO SFU ?
#     # from R-NET part 3.2 gated attention-based recurrent networks
#     with tf.variable_scope("gate"):
#       dim = melt.get_shape(res, -1)
#       d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
#       gate = tf.layers.dense(d_res, dim, use_bias=False, activation=tf.nn.sigmoid, name='gate')
#       return res * gate

# TODO duplicate code  
def dot_attention(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, combiner='gate', scope="dot_attention"):
  with tf.variable_scope(scope):
    d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
    d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
    JX = tf.shape(inputs)[1]
    
    with tf.variable_scope("attention"):
      inputs_ = tf.layers.dense(d_inputs, hidden, use_bias=False, activation=tf.nn.relu, name='inputs')
      memory_ = tf.layers.dense(d_memory, hidden, use_bias=False, activation=tf.nn.relu, name='memory')
      # [batch, input_len, meory_len]
      outputs = tf.matmul(inputs_, tf.transpose(memory_, [0, 2, 1])) / (hidden ** 0.5)
      if mask is not None:
       mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
       logits = tf.nn.softmax(softmax_mask(outputs, mask))
      else:
        logits = tf.nn.softmax(outputs)
      outputs = tf.matmul(logits, memory)

    # for label attention seems gate is better then sfu
    if combiner == 'gate':
      logging.info('dot attention using gate')
      res = tf.concat([inputs, outputs], axis=2)
      with tf.variable_scope("gate"):
        dim = melt.get_shape(res, -1)
        d_res = dropout(res, keep_prob=keep_prob, is_train=is_train)
        gate = tf.layers.dense(d_res, dim, use_bias=False, activation=tf.nn.sigmoid, name='gate')
        return res * gate
    elif combiner == 'sfu' or combiner == 'dsfu':
      logging.info('dot attention using sfu')
      with tf.variable_scope("sfu"):
        if melt.get_shape(outputs, -1) !=  melt.get_shape(inputs, -1):
          outputs = tf.layers.dense(outputs, melt.get_shape(inputs, -1), use_bias=False, activation=None, name='outputs_fc')
        return dsemantic_fusion_combine(inputs, outputs, keep_prob, is_train)
    elif combiner == 'simple_sfu' or combiner == 'simple_dsfu' or combiner == 'ssfu':
      logging.info('dot attention using ssfu')
      with tf.variable_scope("ssfu"):
        if melt.get_shape(outputs, -1) !=  melt.get_shape(inputs, -1):
          outputs = tf.layers.dense(outputs, melt.get_shape(inputs, -1), use_bias=False, activation=None, name='outputs_fc')
        return dsemantic_fusion_simple_combine(inputs, outputs, keep_prob, is_train)     
    else:
      raise ValueError(combiner)

# TODO -------------------------
# just use below really layers!
# TODO batch_dim to batch_axis

keras = tf.keras
layers = tf.keras.layers
Layer = layers.Layer

class MaxPooling(Layer):
  def call(self, outputs, sequence_length=None, axis=1, reduce_func=tf.reduce_max):
    return melt.max_pooling(outputs, sequence_length, axis, reduce_func)

class MaxPooling2(Layer):
  def call(self, outputs, sequence_length, sequence_length2, axis=1, reduce_func=tf.reduce_max):
    return melt.max_pooling2(outputs, sequence_length, sequence_length2, axis, reduce_func)

class MeanPooling(Layer):
  def call(self, outputs, sequence_length=None, axis=1):
    return melt.mean_pooling(outputs, sequence_length, axis)

class FirstPooling(Layer):
  def call(self, outputs, sequence_length=None, axis=1):
    return outputs[:, 0, :]

class LastPooling(Layer):
  def call(self, outputs, sequence_length=None, axis=1):
    return melt.dynamic_last_relevant(outputs, sequence_length)

class HierEncode(Layer):
  def call(self, outputs, sequence_length=None, window_size=3, axis=1):
    return melt.hier_encode(outputs, sequence_length, window_size=3, axis=1)

class TopKPooling(Layer):
  def __init__(self,  
               top_k,
               **kwargs):
    super(TopKPooling, self).__init__(**kwargs)
    self.top_k = top_k

  # def compute_output_shape(self, input_shape):
  #   return (input_shape[0], (input_shape[2] * self.top_k))
  
  def call(self, outputs, sequence_length=None, axis=1):
    x = melt.top_k_pooling(outputs, self.top_k, sequence_length, axis).values  
    #return tf.reshape(x, [melt.get_shape(outputs, 0), -1])
    return tf.reshape(x, [-1, melt.get_shape(outputs, -1) * self.top_k])

def attention_layer(outputs, sequence_length, hidden_size=128, activation=tf.nn.relu, scope='self_attention'):
  with tf.variable_scope('self_attention'):
    hidden_layer = tf.layers.dense(outputs, hidden_size, activation=activation)
    logits = tf.layers.dense(hidden_layer, 1, activation=None)
    alphas = melt.masked_softmax(logits, sequence_length)
    encoding = tf.reduce_sum(outputs * alphas, 1)
    # [batch_size, sequence_length, 1] -> [batch_size, sequence_length]
    alphas = tf.squeeze(alphas) 
    return encoding, alphas

# TODO check which is better tf.nn.tanh or tf.nn.relu, by paper default should be tanh
# TODO check your topk,att cases before use relu.. seems tanh worse then relu, almost eqaul but relu a bit better and stable
# should be keras.Model layer will not save layer...so..
class AttentionPooling(keras.Model):
  def __init__(self,  
               hidden_size=128,
               #activation=tf.nn.tanh,  
               activation=tf.nn.relu,  
               **kwargs):
    super(AttentionPooling, self).__init__(**kwargs)
    self.activation = activation
    if hidden_size is not None:
      self.dense = layers.Dense(hidden_size, activation=activation)
    else:
      self.dense = None
    self.logits = layers.Dense(1, activation=None)
    self.step = -1

  def call(self, outputs, sequence_length=None, axis=1):
    self.step += 1
    if self.step == 0 and self.dense is None:
      self.dense = layers.Dense(melt.get_shape(outputs, -1), activation=self.activation)
    x = self.dense(outputs)
    logits = self.logits(x)
    alphas = tf.nn.softmax(logits) if sequence_length is None else  melt.masked_softmax(logits, sequence_length)
    encoding = tf.reduce_sum(outputs * alphas, 1)
    # [batch_size, sequence_length, 1] -> [batch_size, sequence_length]
    self.alphas = tf.squeeze(alphas, -1)    
    #self.alphas = alphas
    tf.add_to_collection('self_attention', self.alphas) 
    return encoding

class Pooling(keras.Model):
  def __init__(self,  
               name,
               top_k=2,
               **kwargs):
    super(Pooling, self).__init__(**kwargs)

    self.top_k = top_k

    self.poolings = []
    def get_pooling(name):
      if name == 'max':
        return MaxPooling()
      elif name == 'mean':
        return MeanPooling()
      elif name == 'attention' or name == 'att':
        return AttentionPooling()
      elif name == 'attention2' or name == 'att2':
        return AttentionPooling(hidden_size=None)
      elif name == 'topk' or name == 'top_k':
        return TopKPooling(top_k)
      elif name =='first':
        return FirstPooling()
      elif name == 'last':
        return LastPooling()
      else:
        raise f'Unsupport pooling now:{name}'

    self.names = name.split(',')
    for name in self.names:
      self.poolings.append(get_pooling(name))
  
  def call(self, outputs, sequence_length=None, axis=1, calc_word_scores=False):
    results = []
    self.word_scores = []
    for i, pooling in enumerate(self.poolings):
      results.append(pooling(outputs, sequence_length, axis))
      if calc_word_scores:
        self.word_scores.append(melt.get_words_importance(outputs, sequence_length, top_k=self.top_k, method=self.names[i]))
    
    return tf.concat(results, -1)

class DynamicDense(keras.Model):
  def __init__(self,  
               ratio,
               activation=None,
               use_bias=True,
               **kwargs):
    super(DynamicDense, self).__init__(**kwargs)
    self.ratio = ratio  
    self.activation = activation
    self.use_bais = use_bias

    self.step = -1

  def call(self, x):
    self.step += 1
    if self.step == 0:
      self.dense = layers.Dense(melt.get_shape(x, -1) * self.ratio, self.activation, self.use_bais)
    return self.dense(x)

class Embedding(keras.layers.Layer):
  def __init__(self, height, width, name='init_fw'):
    super(Embedding, self).__init__()
    initializer = 'uniform'
    self.embedding = self.add_variable(
        "embedding_kernel",
        shape=[height, width],
        dtype=tf.float32,
        initializer=initializer,
        trainable=True)

  def call(self, x=None):
    if x is not None:
      return tf.nn.embedding_lookup(self.embedding, x)
    else:
      return self.embedding

class Dropout(keras.layers.Layer):
  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
    super(Dropout).__init__(self, **kwargs)
    self.rate = rate 
    self.noise_shape = noise_shape
    self.seed = seed
  
  def call(self, inputs, training=False):
    if not training or self.rate <= 0.:
      return inputs
    else:
      scale = 1.
      shape = tf.shape(inputs)
      if mode == 'embedding':
        noise_shape = [shape[0], 1]
        scale = 1 - self.rate
      elif mode == 'recurrent' and len(inputs.get_shape().as_list()) == 3:
        noise_shape = [shape[0], 1, shape[-1]] 
      return tf.nn.dropout(inputs, 1 - self.rate, noise_shape=noise_shape) * scale

class Gate(keras.Model):
  def __init__(self,
               keep_prob=1.0,
               **kwargs):
    super(Gate, self).__init__(**kwargs)
    self.keep_prob = keep_prob
    self.step = -1

  def call(self, inputs, outputs, training=False):
    self.step += 1
    #with tf.variable_scope(self.scope):
    res = tf.concat([inputs, outputs], axis=2)
    dim = melt.get_shape(res, -1)
    d_res = dropout(res, keep_prob=self.keep_prob, training=training)
    if self.step == 0:
      self.dense = layers.Dense(dim, use_bias=False, activation=tf.nn.sigmoid)
    gate = self.dense(d_res)
    return res * gate

class SemanticFusion(keras.Model):
  def __init__(self,
               keep_prob=1.0,
               **kwargs):
    super(SemanticFusion, self).__init__(**kwargs)
    self.keep_prob = keep_prob
    self.step = -1

  def call(self, input_vector, fusion_vectors, training=False):
    self.step += 1
    assert len(fusion_vectors) > 0
    vectors = tf.concat([input_vector] + fusion_vectors, axis=-1) # size = [batch_size, ..., input_dim * (len(fusion_vectors) + 1)]
    dim = melt.get_shape(input_vector, -1)
    dv = dropout(vectors, keep_prob=self.keep_prob, training=training)
    if self.step == 0:
      self.composition_dense = layers.Dense(dim, use_bias=True, activation=tf.nn.tanh, name='compostion_dense')
      self.gate_dense = layers.Dense(dim, use_bias=True, activation=tf.nn.sigmoid, name='gate_dense')
    r = self.composition_dense(dv)
    g = self.gate_dense(dv)
    return g * r + (1 - g) * input_vector     

class SemanticFusionCombine(keras.Model):
  def __init__(self,
                keep_prob=1.0,
                **kwargs):
      super(SemanticFusionCombine, self).__init__(**kwargs)
      self.keep_prob = keep_prob
      self.sfu = SemanticFusion(keep_prob=keep_prob)
      self.step = -1

  def call(self, x, y, training=False):
    self.step += 1
    if melt.get_shape(x, -1) != melt.get_shape(y, -1):
      if self.step == 0:
        self.dense = layers.Dense(melt.get_shape(x, -1), activation=None, name='sfu_dense')
      y = self.dense(x)
    return self.sfu(x, [y, x * y, x - y], training=training)
  
class DotAttention(keras.Model):
  def __init__(self,
               hidden,
               keep_prob=1.0,
               combiner='gate',
               **kwargs):
    super(DotAttention, self).__init__(**kwargs)
    self.hidden = hidden
    self.keep_prob = keep_prob
    self.combiner = combiner

    self.inputs_dense = layers.Dense(hidden, use_bias=False, activation=tf.nn.relu, name='inputs_dense')
    self.memory_dense = layers.Dense(hidden, use_bias=False, activation=tf.nn.relu, name='memory_dense')

    self.gate = Gate(keep_prob=keep_prob)
    self.sfu = SemanticFusionCombine(keep_prob=keep_prob)

  def call(self, inputs, memory, mask, training=False):
    combiner = self.combiner
    # DotAttention already convert to dot_attention
    #with tf.variable_scope(self.scope):
    d_inputs = dropout(inputs, keep_prob=self.keep_prob, training=training)
    d_memory = dropout(memory, keep_prob=self.keep_prob, training=training)
    JX = tf.shape(inputs)[1]
    
    with tf.variable_scope("attention"):
      inputs_ = self.inputs_dense(d_inputs)
      memory_ = self.memory_dense(d_memory)

      outputs = tf.matmul(inputs_, tf.transpose(memory_, [0, 2, 1])) / (self.hidden ** 0.5)
      if mask is not None:
        mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
        logits = tf.nn.softmax(softmax_mask(outputs, mask))
      else:
        logits = tf.nn.softmax(outputs)

      self.logits = logits
      outputs = tf.matmul(logits, memory)

    if combiner == 'gate':
      return self.gate(inputs, outputs, training=training)
    elif combiner == 'sfu' or combiner == 'dsfu':
      return self.sfu(inputs, outputs, training=training)
    else:
      raise ValueError(combiner)
      
      