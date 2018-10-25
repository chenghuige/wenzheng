#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   rnn.py
#        \author   chenghuige  
#          \date   2016-12-23 14:02:57.513674
#   \Description  
# ==============================================================================

"""
rnn encoding
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

from melt.ops import dynamic_last_relevant, dropout

import copy

import melt
logging = melt.logging

#TODO change from 0, 1 .. to 'forward', 'backward' , 'sum', 'last'
class EncodeMethod:
  forward = 'forward'
  backward = 'backward'
  bidirectional = 'bidirectional'
  stack_bidirectional = 'stack_bidirectional'
  stack_bidirectional_concat_layers = 'stack_bidirectional_concat_layers'
  bidirectional_sum = 'bidirectional_sum'

def is_bidirectional(method):
  return method == EncodeMethod.bidirectional \
      or method == EncodeMethod.stack_bidirectional \
      or method == EncodeMethod.stack_bidirectional_concat_layers \
      or method == EncodeMethod.bidirectional_sum \

class OutputMethod:
  sum = 'sum'
  masked_sum = 'maskedsum'
  last = 'last'
  first = 'first'
  all = 'all'
  mean = 'mean'
  masked_mean = 'maskedmean'
  max = 'max'
  argmax = 'argmax'
  state = 'state'
  attention = 'attention'
  hier = 'hier'

# native gru depreciated, only for low tf like hadoop env
class NativeGru:
  def __init__(self, 
               num_layers, 
               num_units, 
               keep_prob=1.0, 
               share_dropout=True,
               dropout_mode=None, 
               train_init_state=True,
               is_train=None, 
               scope="native_gru"):
    self.num_layers = num_layers
    self.keep_prob = keep_prob
    self.num_units = num_units
    self.is_train = is_train
    self.train_init_state = train_init_state
    self.scope = scope

    self.share_dropout = share_dropout
    self.dropout_mode = dropout_mode
    self.dropout_mask_fw = [None] * num_layers
    self.dropout_mask_bw = [None] * num_layers

    self.init_fw = [None] * num_layers
    self.init_bw = [None] * num_layers 

  def set_dropout_mask(self, mask_fw, mask_bw):
    self.dropout_mask_fw = mask_fw 
    self.dropout_mask_bw = mask_bw

  def set_init_states(self, init_fw, init_bw):
    self.init_fw = init_fw
    self.init_bw = init_bw

  def reset_init_states(self):
    self.init_fw = [None] * self.num_layers
    self.init_bw = [None] * self.num_layers     

  def encode(self, inputs, seq_len, emb=None, concat_layers=True, output_method=OutputMethod.all):
    if emb is not None:
      inputs = tf.nn.embedding_lookup(emb, inputs)

    outputs = [inputs]
    keep_prob = self.keep_prob
    num_units = self.num_units
    is_train = self.is_train

    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      for layer in range(self.num_layers):
        input_size_ = melt.get_shape(inputs, -1) if layer == 0 else 2 * num_units
        batch_size = melt.get_batch_size(inputs)
        with tf.variable_scope("fw_{}".format(layer)):
          gru_fw = tf.contrib.rnn.GRUCell(num_units)
          if not self.share_dropout:
            mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                                       keep_prob=keep_prob, is_train=is_train, mode=self.dropout_mode)
          else:
            if self.dropout_mask_fw[layer] is None:
              mask_fw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                                         keep_prob=keep_prob, is_train=is_train, mode=self.dropout_mode)
              self.dropout_mask_fw[layer] = mask_fw
            else:
              mask_fw = self.dropout_mask_fw[layer]
          if self.train_init_state:
            if self.init_fw[layer] is None:
              self.init_fw[layer] = tf.tile(tf.get_variable("init_state", [1, num_units], tf.float32, tf.zeros_initializer()), [batch_size, 1])
          out_fw, state = tf.nn.dynamic_rnn(
                            gru_fw, outputs[-1] * mask_fw, seq_len, initial_state=self.init_fw[layer], dtype=tf.float32)
        with tf.variable_scope("bw_{}".format(layer)):
          gru_bw = tf.contrib.rnn.GRUCell(num_units)
          if not self.share_dropout:
            mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                                       keep_prob=keep_prob, is_train=is_train, mode=self.dropout_mode)           
          else:
            if self.dropout_mask_bw[layer] is None:
              mask_bw = dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                                         keep_prob=keep_prob, is_train=is_train, mode=self.dropout_mode)
              self.dropout_mask_bw[layer] = mask_bw
            else:
              mask_bw = self.dropout_mask_bw[layer]
          if self.train_init_state:
            if self.init_bw[layer] is None:
              self.init_bw[layer] = tf.tile(tf.get_variable("init_state", [1, num_units], tf.float32, tf.zeros_initializer()), [batch_size, 1])
          inputs_bw = tf.reverse_sequence(
                        outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
          out_bw, _ = tf.nn.dynamic_rnn(
                        gru_bw, inputs_bw, seq_len, initial_state=self.init_bw[layer], dtype=tf.float32)
          out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
        outputs.append(tf.concat([out_fw, out_bw], axis=2))
    
    if concat_layers:
      res = tf.concat(outputs[1:], axis=2)
    else:
      res = outputs[-1]
    res = encode_outputs(res, seq_len, output_method=output_method)
    return res

  def __call__(self, inputs, seq_len, emb=None, concat_layers=True, output_method=OutputMethod.all):
    return self.encode(inputs, seq_len, emb, concat_layers, output_method)


class CudnnRnn:
  def __init__(self, 
                cell, 
                num_layers, 
                num_units, 
                keep_prob=1.0, 
                share_dropout=True,
                train_init_state=True,
                is_train=None, 
                scope='cudnn'):
    self.cell = cell
    if isinstance(cell, str):
      if cell == 'gru':
        self.cell = tf.contrib.cudnn_rnn.CudnnGRU
        scope = scope + '_gru'
      elif cell == 'lstm':
        self.cell = tf.contrib.cudnn_rnn.CudnnLSTM 
        scope = scope + '_lstm'
      else:
        raise ValueError(cell)

    logging.info('cudnn cell:', self.cell)
    self.num_layers = num_layers
    self.keep_prob = keep_prob
    assert num_units % 4 == 0, 'bad performance for units size not % 4'
    self.num_units = num_units
    self.is_train = is_train
    self.scope=scope

    # for share dropout between like context and question in squad (machine reading task)
    # rnn = gru(num_layers=FLAGS.num_layers, num_units=d, keep_prob=keep_prob, is_train=self.is_training)
    # c = rnn(c_emb, seq_len=c_len)
    # scope.reuse_variables()
    # q = rnn(q_emb, seq_len=q_len)
    self.share_dropout = share_dropout
    self.dropout_mask_fw = [None] * num_layers
    self.dropout_mask_bw = [None] * num_layers 

    self.train_init_state = train_init_state
    self.init_fw = [None] * num_layers
    self.init_bw = [None] * num_layers 

    self.state = None

  def set_dropout_mask(self, mask_fw, mask_bw):
    self.dropout_mask_fw = mask_fw 
    self.dropout_mask_bw = mask_bw

  def set_init_states(self, init_fw, init_bw):
    self.init_fw = init_fw
    self.init_bw = init_bw

  def reset_init_states(self):
    self.init_fw = [None] * self.num_layers
    self.init_bw = [None] * self.num_layers     

  def encode(self, inputs, seq_len, emb=None, concat_layers=True, output_method=OutputMethod.all):
    if emb is not None:
      inputs = tf.nn.embedding_lookup(emb, inputs)
      
    outputs = [tf.transpose(inputs, [1, 0, 2])]
    #states = []
    keep_prob = self.keep_prob
    num_units = self.num_units
    is_train = self.is_train

    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      for layer in range(self.num_layers):
        input_size_ = melt.get_shape(inputs, -1) if layer == 0 else 2 * num_units
        batch_size = melt.get_batch_size(inputs)

        with tf.variable_scope("fw_{}".format(layer)):
          gru_fw = self.cell(num_layers=1, num_units=num_units)
          if not self.share_dropout:
            mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                          keep_prob=keep_prob, is_train=is_train, mode=None)
          else:
            if self.dropout_mask_fw[layer] is None:
              mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                                          keep_prob=keep_prob, is_train=is_train, mode=None)
              self.dropout_mask_fw[layer] = mask_fw
            else:
              mask_fw = self.dropout_mask_fw[layer]
          if self.train_init_state:
            if self.init_fw[layer] is None:
              self.init_fw[layer] = (tf.tile(tf.get_variable("init_state", [1, 1, num_units], tf.float32, tf.zeros_initializer()), [1, batch_size, 1]),)
          out_fw, state_fw = gru_fw(outputs[-1] * mask_fw, self.init_fw[layer])

        with tf.variable_scope("bw_{}".format(layer)):
          gru_bw = self.cell(num_layers=1, num_units=num_units)
          if not self.share_dropout:
            mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                                        keep_prob=keep_prob, is_train=is_train, mode=None)
          else:
            if self.dropout_mask_bw[layer] is None:
              mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                                          keep_prob=keep_prob, is_train=is_train, mode=None)
              self.dropout_mask_bw[layer] = mask_bw
            else:
              mask_bw = self.dropout_mask_bw[layer]
          inputs_bw = tf.reverse_sequence(
              outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
          if self.train_init_state:
            if self.init_bw[layer] is None:
              self.init_bw[layer] = (tf.tile(tf.get_variable("init_state", [1, 1, num_units], tf.float32, tf.zeros_initializer()), [1, batch_size, 1]),)
          out_bw, state_bw = gru_bw(inputs_bw, self.init_bw[layer])
          out_bw = tf.reverse_sequence(
              out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)

        outputs.append(tf.concat([out_fw, out_bw], axis=2))
        #states.append(tf.concat([state_fw, state_bw], axis=-1))

    if concat_layers:
      res = tf.concat(outputs[1:], axis=2)
      #state = tf.concat(states, axis=-1)
    else:
      res = outputs[-1]
      #state = states[-1]

    res = tf.transpose(res, [1, 0, 2])
    #state = tf.squeeze(state)
    #state = tf.reshape(state, [-1, num_units * 2 * self.num_layers])
    #res = encode_outputs(res, output_method=output_method, sequence_length=seq_len, state=state)
    res = encode_outputs(res, output_method=output_method, sequence_length=seq_len)

    self.state = (state_fw, state_bw)

    return res

  def __call__(self, inputs, seq_len, emb=None, concat_layers=True, output_method=OutputMethod.all):
    return self.encode(inputs, seq_len, emb, concat_layers, output_method)

# tested on tf1.6
class CudnnGru:
  def __init__(self, 
                num_layers, 
                num_units, 
                keep_prob=1.0, 
                share_dropout=True, 
                train_init_state=True, 
                is_train=None, 
                scope='cudnn_gru'):
    self.num_layers = num_layers
    self.keep_prob = keep_prob
    assert num_units % 4 == 0, 'bad performance for units size not % 4'
    self.num_units = num_units
    self.is_train = is_train
    self.scope=scope

    self.share_dropout = share_dropout
    self.dropout_mask_fw = [None] * num_layers
    self.dropout_mask_bw = [None] * num_layers 

    self.train_init_state = train_init_state
    self.init_fw = [None] * num_layers
    self.init_bw = [None] * num_layers 

    self.state = None

  def set_dropout_mask(self, mask_fw, mask_bw):
    self.dropout_mask_fw = mask_fw 
    self.dropout_mask_bw = mask_bw

  def set_init_states(self, init_fw, init_bw):
    self.init_fw = init_fw
    self.init_bw = init_bw

  def reset_init_states(self):
    self.init_fw = [None] * self.num_layers
    self.init_bw = [None] * self.num_layers      

  def encode(self, inputs, seq_len, emb=None, concat_layers=True, output_method=OutputMethod.all):
    if emb is not None:
      inputs = tf.nn.embedding_lookup(emb, inputs)
      
    outputs = [tf.transpose(inputs, [1, 0, 2])]
    #states = []
    keep_prob = self.keep_prob
    num_units = self.num_units
    is_train = self.is_train

    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      for layer in range(self.num_layers):
        input_size_ = melt.get_shape(inputs, -1) if layer == 0 else 2 * num_units
        batch_size = melt.get_batch_size(inputs)

        with tf.variable_scope("fw_{}".format(layer)):
          gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1, num_units=num_units)
          if not self.share_dropout:
            # mode is None since by define mask.. is already recurrent mode
            mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                                        keep_prob=keep_prob, is_train=is_train, mode=None)
          else:             
            if self.dropout_mask_fw[layer] is None:
              mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                                          keep_prob=keep_prob, is_train=is_train, mode=None)
              self.dropout_mask_fw[layer] = mask_fw
            else:
              mask_fw = self.dropout_mask_fw[layer]
          if self.train_init_state:
            if self.init_fw[layer] is None:
              self.init_fw[layer] = (tf.tile(tf.get_variable("init_state", [1, 1, num_units], tf.float32, tf.zeros_initializer()), [1, batch_size, 1]),)
          out_fw, state_fw = gru_fw(outputs[-1] * mask_fw, self.init_fw[layer])

        with tf.variable_scope("bw_{}".format(layer)):
          gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1, num_units=num_units)
          if not self.share_dropout:
            mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                                        keep_prob=keep_prob, is_train=is_train, mode=None)
          else:              
            if self.dropout_mask_bw[layer] is None:
              mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                                          keep_prob=keep_prob, is_train=is_train, mode=None)
              self.dropout_mask_bw[layer] = mask_bw
            else:
              mask_bw = self.dropout_mask_bw[layer]
          inputs_bw = tf.reverse_sequence(
              outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
          if self.train_init_state:
            if self.init_bw[layer] is None:
              self.init_bw[layer] = (tf.tile(tf.get_variable("init_state", [1, 1, num_units], tf.float32, tf.zeros_initializer()), [1, batch_size, 1]),)
          out_bw, state_bw = gru_bw(inputs_bw, self.init_bw[layer])
          out_bw = tf.reverse_sequence(
              out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
        
        outputs.append(tf.concat([out_fw, out_bw], axis=2))
        #states.append(tf.concat([state_fw, state_bw], axis=-1))

    if concat_layers:
      res = tf.concat(outputs[1:], axis=2)
      #state = tf.concat(states, axis=-1)
    else:
      res = outputs[-1]
      #state = states[-1]

    res = tf.transpose(res, [1, 0, 2])
    #state = tf.squeeze(state)
    #state = tf.reshape(state, [-1, num_units * 2 * self.num_layers])
    #res = encode_outputs(res, output_method=output_method, sequence_length=seq_len, state=state)
    res = encode_outputs(res, output_method=output_method, sequence_length=seq_len)

    self.state = (state_fw, state_bw)
    return res

  def __call__(self, inputs, seq_len, emb=None, concat_layers=True, output_method=OutputMethod.all):
    return self.encode(inputs, seq_len, emb, concat_layers, output_method)

# else:
#   logging.warning('using native_gru instead of cudnn due to low tf version then 1.5')
#   CudnnGru = NativeGru

class NullEncoder():
  def encode(self, inputs, sequence_length, output_method='all'):
    return encode_outputs(inputs, sequence_length, output_method)

# TODO move from rnn to general lib like ops or layers         
# TODO add support for top_k in addtion to max pooling 
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52526#299619
# #     def _top_k(x):
#         x = tf.transpose(x, [0, 2, 1])
#         k_max = tf.nn.top_k(x, k=top_k)
#         return tf.reshape(k_max[0], (-1, 2 * num_filters * top_k))                                                                                                                                                                                                                                                                     
def encode_outputs(outputs, sequence_length=None, 
                   output_method=OutputMethod.last,  
                   state=None,
                   attention_hidden_size=128,
                   window_size=3):
  if output_method == OutputMethod.state:
    assert state is not None
    return state 

  #--seems slower convergence and not good result when only using last output, so change to use sum
  if output_method == OutputMethod.sum:
    return tf.reduce_sum(outputs, 1)
  elif output_method == OutputMethod.masked_sum:
    return melt.sum_pooling(outputs, sequence_length)
  elif output_method == OutputMethod.max:
    assert sequence_length is not None
    #below not work.. sequence is different for each row instance
    #return tf.reduce_max(outputs[:, :sequence_length, :], 1)
    #return tf.reduce_max(outputs, 1) #not exclude padding embeddings
    #return tf.reduce_max(tf.abs(outputs), 1)
    return melt.max_pooling(outputs, sequence_length)
  elif output_method == OutputMethod.argmax:
    assert sequence_length is not None
    #return tf.argmax(outputs[:, :sequence_length, :], 1)
    #return tf.argmax(outputs, 1)
    #return tf.argmax(tf.abs(outputs), 1)
    return melt.argmax_pooling(outputs, sequence_length)
  elif output_method == OutputMethod.mean:
    assert sequence_length is not None
    return tf.reduce_sum(outputs, 1) / tf.to_float(tf.expand_dims(sequence_length, 1)) 
  elif output_method == OutputMethod.masked_mean:
    return melt.mean_pooling(outputs, sequence_length)
  elif output_method == OutputMethod.last:
    #TODO actually return state.h is last revlevant?
    return dynamic_last_relevant(outputs, sequence_length)
  elif output_method == OutputMethod.first:
    return outputs[:, 0, :]
  elif output_method == OutputMethod.attention:
    logging.info('attention_hidden_size:', attention_hidden_size)
    encoding, alphas = melt.layers.self_attention(outputs, sequence_length, attention_hidden_size)
    tf.add_to_collection('self_attention', alphas)
    return encoding
  elif output_method == OutputMethod.hier:
    return melt.hier_pooling(outputs, sequence_length, window_size=window_size)
  else: # all
    return outputs

def forward_encode(cell, inputs, sequence_length, initial_state=None, dtype=None, output_method=OutputMethod.last):
  outputs, state = tf.nn.dynamic_rnn(
    cell, 
    inputs, 
    initial_state=initial_state, 
    dtype=dtype,
    sequence_length=sequence_length)
  
  return encode_outputs(outputs, sequence_length, output_method), state


def backward_encode(cell, inputs, sequence_length, initial_state=None, dtype=None, output_method=OutputMethod.last):
  outputs, state = tf.nn.dynamic_rnn(
    cell, 
    tf.reverse_sequence(inputs, sequence_length, 1), 
    initial_state=initial_state, 
    dtype=dtype,
    sequence_length=sequence_length)

  return encode_outputs(outputs, sequence_length, output_method), state

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope as vs
def stack_bidirectional_dynamic_rnn(cells_fw,
                                    cells_bw,
                                    inputs,
                                    initial_states_fw=None,
                                    initial_states_bw=None,
                                    dtype=None,
                                    sequence_length=None,
                                    parallel_iterations=None,
                                    time_major=False,
                                    scope=None):
  """Creates a dynamic bidirectional recurrent neural network.

  Stacks several bidirectional rnn layers. The combined forward and backward
  layer outputs are used as input of the next layer. tf.bidirectional_rnn
  does not allow to share forward and backward information between layers.
  The input_size of the first forward and backward cells must match.
  The initial state for both directions is zero and no intermediate states
  are returned.

  Args:
    cells_fw: List of instances of RNNCell, one per layer,
      to be used for forward direction.
    cells_bw: List of instances of RNNCell, one per layer,
      to be used for backward direction.
    inputs: The RNN inputs. this must be a tensor of shape:
      `[batch_size, max_time, ...]`, or a nested tuple of such elements.
    initial_states_fw: (optional) A list of the initial states (one per layer)
      for the forward RNN.
      Each tensor must has an appropriate type and shape
      `[batch_size, cell_fw.state_size]`.
    initial_states_bw: (optional) Same as for `initial_states_fw`, but using
      the corresponding properties of `cells_bw`.
    dtype: (optional) The data type for the initial state.  Required if
      either of the initial states are not provided.
    sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
      containing the actual lengths for each of the sequences.
    parallel_iterations: (Default: 32).  The number of iterations to run in
      parallel.  Those operations which do not have any temporal dependency
      and can be run in parallel, will be.  This parameter trades off
      time for space.  Values >> 1 use more memory but take less time,
      while smaller values use less memory but computations take longer.
    time_major: The shape format of the inputs and outputs Tensors. If true,
      these Tensors must be shaped [max_time, batch_size, depth]. If false,
      these Tensors must be shaped [batch_size, max_time, depth]. Using
      time_major = True is a bit more efficient because it avoids transposes at
      the beginning and end of the RNN calculation. However, most TensorFlow
      data is batch-major, so by default this function accepts input and emits
      output in batch-major form.
    scope: VariableScope for the created subgraph; defaults to None.

  Returns:
    A tuple (outputs, output_state_fw, output_state_bw) where:
      outputs: Output `Tensor` shaped:
        `batch_size, max_time, layers_output]`. Where layers_output
        are depth-concatenated forward and backward outputs.
      output_states_fw is the final states, one tensor per layer,
        of the forward rnn.
      output_states_bw is the final states, one tensor per layer,
        of the backward rnn.

  Raises:
    TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
    ValueError: If inputs is `None`.
  """
  if not cells_fw:
    raise ValueError("Must specify at least one fw cell for BidirectionalRNN.")
  if not cells_bw:
    raise ValueError("Must specify at least one bw cell for BidirectionalRNN.")
  if not isinstance(cells_fw, list):
    raise ValueError("cells_fw must be a list of RNNCells (one per layer).")
  if not isinstance(cells_bw, list):
    raise ValueError("cells_bw must be a list of RNNCells (one per layer).")
  if len(cells_fw) != len(cells_bw):
    raise ValueError("Forward and Backward cells must have the same depth.")
  if (initial_states_fw is not None and
      (not isinstance(initial_states_fw, list) or
       len(initial_states_fw) != len(cells_fw))):
    raise ValueError(
        "initial_states_fw must be a list of state tensors (one per layer).")
  if (initial_states_bw is not None and
      (not isinstance(initial_states_bw, list) or
       len(initial_states_bw) != len(cells_bw))):
    raise ValueError(
        "initial_states_bw must be a list of state tensors (one per layer).")

  states_fw = []
  states_bw = []
  prev_layer = inputs

  outputs_list = [prev_layer]

  with vs.variable_scope(scope or "stack_bidirectional_rnn"):
    for i, (cell_fw, cell_bw) in enumerate(zip(cells_fw, cells_bw)):
      initial_state_fw = None
      initial_state_bw = None
      if initial_states_fw:
        initial_state_fw = initial_states_fw[i]
      if initial_states_bw:
        initial_state_bw = initial_states_bw[i]

      with vs.variable_scope("cell_%d" % i):
        outputs, (state_fw, state_bw) = rnn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            prev_layer,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            sequence_length=sequence_length,
            parallel_iterations=parallel_iterations,
            dtype=dtype,
            time_major=time_major)
        # Concat the outputs to create the new input.
        prev_layer = array_ops.concat(outputs, 2)
        outputs_list.append(prev_layer)
      states_fw.append(state_fw)
      states_bw.append(state_bw)

  return tf.concat(outputs_list[1:], -1), tuple(states_fw), tuple(states_bw)

def bidirectional_encode(cell_fw, 
                        cell_bw, 
                        inputs, 
                        sequence_length, 
                        initial_state_fw=None, 
                        initial_state_bw=None, 
                        dtype=None,
                        output_method=OutputMethod.last,
                        use_sum=False,
                        is_stack=False,
                        is_stack_concat_layers=False):
  # if cell_bw is None:
  #   cell_bw = copy.deepcopy(cell_fw)
  assert cell_fw is not None 
  assert cell_bw is not None
  if initial_state_bw is None:
    initial_state_bw = initial_state_fw

  if melt.is_cudnn_cell(cell_fw):
    outputs = [tf.transpose(inputs, [1, 0, 2])]
    for layer in range(cell_fw.num_layers):
      with tf.variable_scope("fw_{}".format(layer)):
        out_fw, state_fw = cell_fw(outputs[-1])
      with tf.variable_scope("bw_{}".format(layer)):
        inputs_bw = tf.reverse_sequence(
          outputs[-1], seq_lengths=sequence_length, seq_dim=0, batch_dim=1)
        out_bw, state_bw = cell_bw(inputs_bw)
        out_bw = tf.reverse_sequence(
            out_bw, seq_lengths=sequence_length, seq_dim=0, batch_dim=1)
    outputs.append(tf.concat([out_fw, out_bw], axis=2))
    res = tf.concat(outputs[1:], axis=2)
    print('-------------res, state_fw', res, state_fw)
    #print(tf.split(state_fw, cell_fw.num_layers))
    #return res, tf.split(state_fw, cell_fw.num_layers)
    return res, state_fw

  if not is_stack:
    outputs, states  = tf.nn.bidirectional_dynamic_rnn(
      cell_fw=cell_fw,
      cell_bw=cell_bw,
      inputs=inputs,
      initial_state_fw=initial_state_fw,
      initial_state_bw=initial_state_bw,
      dtype=dtype,
      sequence_length=sequence_length)
    output_fws, output_bws = outputs
    output_forward = encode_outputs(output_fws, sequence_length, output_method)
    output_backward = encode_outputs(output_bws, sequence_length, output_method)

    if output_method == OutputMethod.sum:
      output_backward = tf.reduce_sum(output_bws, 1) 

    if use_sum:
      output = output_forward + output_backward
    else:
      output = tf.concat([output_forward, output_backward], -1)
  else:
    if not is_stack_concat_layers:
      func = tf.contrib.rnn.stack_bidirectional_dynamic_rnn
    else:
      func = stack_bidirectional_dynamic_rnn
    output, states_fw, states_bw = func(
      cells_fw=melt.unpack_cell(cell_fw),
      cells_bw=melt.unpack_cell(cell_bw),
      inputs=inputs,
      initial_states_fw=initial_state_fw,
      initial_states_bw=initial_state_bw,
      dtype=dtype,
      sequence_length=sequence_length)
    states = (states_fw, states_bw)
    output = encode_outputs(output, sequence_length, output_method)
      
  #TODO state[0] ?
  return output, states[0]

def encode(cell, 
           inputs, 
           sequence_length=None, 
           initial_state=None, 
           cell_bw=None, 
           inital_state_bw=None, 
           dtype=None,
           encode_method=EncodeMethod.forward, 
           output_method=OutputMethod.last):
    assert sequence_length is not None, 'bidrecional encoding need seq len, for safe all pass sequence_length !'

    #needed for bidirectional_dynamic_rnn and backward method
    #without it Input 'seq_lengths' of 'ReverseSequence' Op has type int32 that does not match expected type of int64.
    #int tf.reverse_sequence seq_lengths: A `Tensor` of type `int64`.
    if initial_state is None and dtype is None:
      dtype = tf.float32
    sequence_length = tf.cast(sequence_length, tf.int64)
    if encode_method == EncodeMethod.forward:
      return forward_encode(cell, inputs, sequence_length, initial_state, dtype, output_method)
    elif encode_method == EncodeMethod.backward:
      return backward_encode(cell, inputs, sequence_length, initial_state, dtype, output_method)
    elif encode_method == EncodeMethod.bidirectional:
      return bidirectional_encode(cell, cell_bw, inputs, sequence_length, 
                                 initial_state, inital_state_bw, dtype, output_method)
    elif encode_method == EncodeMethod.stack_bidirectional:
      return bidirectional_encode(cell, cell_bw, inputs, sequence_length, 
                                 initial_state, inital_state_bw, dtype, output_method, 
                                 is_stack=True)
    elif encode_method == EncodeMethod.stack_bidirectional_concat_layers:
      return bidirectional_encode(cell, cell_bw, inputs, sequence_length, 
                                 initial_state, inital_state_bw, dtype, output_method, 
                                 is_stack=True, is_stack_concat_layers=True)
    elif encode_method == EncodeMethod.bidirectional_sum:
      return bidirectional_encode(cell, cell_bw, inputs, sequence_length, 
                                 initial_state, inital_state_bw, dtype, output_method,
                                 use_sum=True)
    else:
      raise ValueError('Unsupported rnn encode method:', encode_method)
