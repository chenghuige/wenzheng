    # Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Attention-based decoder functions. 
depreciated using attention_wrapper instead
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.ops import tensor_array_ops

__all__ = [
    "attention_decoder_fn_train",
    "attention_decoder_fn_inference",
    "init_attention"
]

import tensorflow as tf
import melt

def attention_decoder_fn_train(encoder_state,
                               attention_keys,
                               attention_values,
                               attention_score_fn,
                               attention_construct_fn,
                               input_text=None,
                               vocab_size=None,
                               name=None):
  """Attentional decoder function for `dynamic_rnn_decoder` during training.

  The `attention_decoder_fn_train` is a training function for an
  attention-based sequence-to-sequence model. It should be used when
  `dynamic_rnn_decoder` is in the training mode.

  The `attention_decoder_fn_train` is called with a set of the user arguments
  and returns the `decoder_fn`, which can be passed to the
  `dynamic_rnn_decoder`, such that

  ```
  dynamic_fn_train = attention_decoder_fn_train(encoder_state)
  outputs_train, state_train = dynamic_rnn_decoder(
      decoder_fn=dynamic_fn_train, ...)
  ```

  Further usage can be found in the `kernel_tests/seq2seq_test.py`.

  Args:
    encoder_state: The encoded state to initialize the `dynamic_rnn_decoder`.
    attention_keys: to be compared with target states.
    attention_values: to be used to construct context vectors.
    attention_score_fn: to compute similarity between key and target states.
    attention_construct_fn: to build attention states.
    name: (default: `None`) NameScope for the decoder function;
      defaults to "simple_decoder_fn_train"

  Returns:
    A decoder function with the required interface of `dynamic_rnn_decoder`
    intended for training.
  """
  with ops.name_scope(name, "attention_decoder_fn_train", [
      encoder_state, attention_keys, attention_values, attention_score_fn,
      attention_construct_fn
  ]):
    pass

  def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
    """Decoder function used in the `dynamic_rnn_decoder` for training.

    Args:
      time: positive integer constant reflecting the current timestep.
      cell_state: state of RNNCell.
      cell_input: input provided by `dynamic_rnn_decoder`.
      cell_output: output of RNNCell.
      context_state: context state provided by `dynamic_rnn_decoder`.

    Returns:
      A tuple (done, next state, next input, emit output, next context state)
      where:

      done: `None`, which is used by the `dynamic_rnn_decoder` to indicate
      that `sequence_lengths` in `dynamic_rnn_decoder` should be used.

      next state: `cell_state`, this decoder function does not modify the
      given state.

      next input: `cell_input`, this decoder function does not modify the
      given input. The input could be modified when applying e.g. attention.

      emit output: `cell_output`, this decoder function does not modify the
      given output.

      next context state: `context_state`, this decoder function does not
      modify the given context state. The context state could be modified when
      applying e.g. beam search.
    """
    with ops.name_scope(
        name, "attention_decoder_fn_train",
        [time, cell_state, cell_input, cell_output, context_state]):
      #input_text = None
      
      if cell_state is None:  # first call, return encoder_state
        cell_state = encoder_state

        # init attention
        attention = init_attention(encoder_state)
        if input_text is not None:
          #cell_output = array_ops.zeros([vocab_size], dtype=dtypes.float32)
          context_state = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=0, dynamic_size=True, infer_shape=False)
      else:
        # construct attention
        attention, scores, alignments = attention_construct_fn(cell_output, attention_keys,
                                                               attention_values)
        #attention, scores, alignments = attention_construct_fn(cell_state.h, attention_keys,
        #                                                       attention_values)
        cell_output = attention
        if input_text is not None:
          #encoder_info = nest.flatten(encoder_state)[0]
          #batch_size = encoder_info.get_shape()[0].value
          #if batch_size is None:
          #  batch_size = array_ops.shape(encoder_info)[0]
          ##ref =  tf.Variable(array_ops.zeros([batch_size, vocab_size], dtype=dtypes.float32))
          ##https://github.com/tensorflow/tensorflow/issues/8604
          #with tf.control_dependencies(None):
            ##TODO... must fix batch_size right now
          #  cell_output_ =  tf.Variable(array_ops.zeros([256, vocab_size], dtype=dtypes.float32))

          #actually should use tf.scatter_nd, anyway this attention is deprecated!
          cell_output_ =  tf.get_variable("cell_output_", [256, vocab_size], dtype=dtypes.float32, initializer=tf.zeros_initializer())
          #cell_output_ =  tf.get_variable("cell_output_", [256, vocab_size], dtype=dtypes.float32, initializer=tf.ones_initializer())
          #cell_output_.assign(array_ops.zeros([256, vocab_size]))
          cell_output_ = tf.assign(cell_output_, array_ops.zeros([256, vocab_size]))
          indices = melt.batch_values_to_indices(tf.to_int32(input_text))
          updates = scores
          #updates = alignments
          cell_output_ = tf.scatter_nd_add(cell_output_, indices, updates)
          cell_output_ = tf.stop_gradient(cell_output_)
          ##print(cell_output, vocab_size)
          ##cell_output = attention
          ##cell_output = array_ops.zeros([batch_size, vocab_size], dtype=dtypes.float32)
          ##cell_output = scores
          #cell_output = tf.convert_to_tensor(cell_output)

          context_state = context_state.write(time - 1, cell_output_)
          #context_state = context_state.write(time - 1, scores)

      # combine cell_input and attention
      next_input = array_ops.concat([cell_input, attention], 1)
      #next_input = cell_input

      return (None, cell_state, next_input, cell_output, context_state)

  return decoder_fn

def attention_decoder_fn_inference(output_fn,
                                   first_input,
                                   encoder_state,
                                   attention_keys,
                                   attention_values,
                                   attention_score_fn,
                                   attention_construct_fn,
                                   embeddings,
                                   end_of_sequence_id,
                                   maximum_length,
                                   num_decoder_symbols,
                                   input_text=None,
                                   dtype=dtypes.int32,
                                   name=None):
  """Attentional decoder function for `dynamic_rnn_decoder` during inference.
  The `attention_decoder_fn_inference` is a simple inference function for a
  sequence-to-sequence model. It should be used when `dynamic_rnn_decoder` is
  in the inference mode.
  The `attention_decoder_fn_inference` is called with user arguments
  and returns the `decoder_fn`, which can be passed to the
  `dynamic_rnn_decoder`, such that
  ```
  dynamic_fn_inference = attention_decoder_fn_inference(...)
  outputs_inference, state_inference = dynamic_rnn_decoder(
      decoder_fn=dynamic_fn_inference, ...)
  ```
  Further usage can be found in the `kernel_tests/seq2seq_test.py`.
  Args:
    output_fn: An output function to project your `cell_output` onto class
    logits.
    An example of an output function;
    ```
      tf.variable_scope("decoder") as varscope
        output_fn = lambda x: layers.linear(x, num_decoder_symbols,
                                            scope=varscope)
        outputs_train, state_train = seq2seq.dynamic_rnn_decoder(...)
        logits_train = output_fn(outputs_train)
        varscope.reuse_variables()
        logits_inference, state_inference = seq2seq.dynamic_rnn_decoder(
            output_fn=output_fn, ...)
    ```
    If `None` is supplied it will act as an identity function, which
    might be wanted when using the RNNCell `OutputProjectionWrapper`.
    encoder_state: The encoded state to initialize the `dynamic_rnn_decoder`.
    attention_keys: to be compared with target states.
    attention_values: to be used to construct context vectors.
    attention_score_fn: to compute similarity between key and target states.
    attention_construct_fn: to build attention states.
    embeddings: The embeddings matrix used for the decoder sized
    `[num_decoder_symbols, embedding_size]`.
    start_of_sequence_id: The start of sequence ID in the decoder embeddings.
    end_of_sequence_id: The end of sequence ID in the decoder embeddings.
    maximum_length: The maximum allowed of time steps to decode.
    num_decoder_symbols: The number of classes to decode at each time step.
    dtype: (default: `dtypes.int32`) The default data type to use when
    handling integer objects.
    name: (default: `None`) NameScope for the decoder function;
      defaults to "attention_decoder_fn_inference"
  Returns:
    A decoder function with the required interface of `dynamic_rnn_decoder`
    intended for inference.
  """
  with ops.name_scope(name, "attention_decoder_fn_inference", [
      output_fn, first_input, encoder_state, 
      attention_keys, attention_values,
      attention_score_fn, attention_construct_fn, embeddings,
      end_of_sequence_id, maximum_length,
      num_decoder_symbols, dtype
  ]):
    first_input = ops.convert_to_tensor(first_input, dtypes.float32)
    end_of_sequence_id = ops.convert_to_tensor(end_of_sequence_id, dtype)
    maximum_length = ops.convert_to_tensor(maximum_length, dtype)
    num_decoder_symbols = ops.convert_to_tensor(num_decoder_symbols, dtype)
    encoder_info = nest.flatten(encoder_state)[0]
    batch_size = encoder_info.get_shape()[0].value
    if output_fn is None:
      output_fn = lambda x: x
    if batch_size is None:
      batch_size = array_ops.shape(encoder_info)[0]

  def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
    """Decoder function used in the `dynamic_rnn_decoder` for inference.
    The main difference between this decoder function and the `decoder_fn` in
    `attention_decoder_fn_train` is how `next_cell_input` is calculated. In
    decoder function we calculate the next input by applying an argmax across
    the feature dimension of the output from the decoder. This is a
    greedy-search approach. (Bahdanau et al., 2014) & (Sutskever et al., 2014)
    use beam-search instead.
    Args:
      time: positive integer constant reflecting the current timestep.
      cell_state: state of RNNCell.
      cell_input: input provided by `dynamic_rnn_decoder`.
      cell_output: output of RNNCell.
      context_state: context state provided by `dynamic_rnn_decoder`.
    Returns:
      A tuple (done, next state, next input, emit output, next context state)
      where:
      done: A boolean vector to indicate which sentences has reached a
      `end_of_sequence_id`. This is used for early stopping by the
      `dynamic_rnn_decoder`. When `time>=maximum_length` a boolean vector with
      all elements as `true` is returned.
      next state: `cell_state`, this decoder function does not modify the
      given state.
      next input: The embedding from argmax of the `cell_output` is used as
      `next_input`.
      emit output: If `output_fn is None` the supplied `cell_output` is
      returned, else the `output_fn` is used to update the `cell_output`
      before calculating `next_input` and returning `cell_output`.
      next context state: `context_state`, this decoder function does not
      modify the given context state. The context state could be modified when
      applying e.g. beam search.
    Raises:
      ValueError: if cell_input is not None.
    """
    with ops.name_scope(
        name, "attention_decoder_fn_inference",
        [time, cell_state, cell_input, cell_output, context_state]):
      if cell_input is not None:
        raise ValueError("Expected cell_input to be None, but saw: %s" %
                         cell_input)
      if cell_output is None:
        # invariant that this is time == 0
        next_input_id = None
        done = array_ops.zeros([batch_size,], dtype=dtypes.bool)
        cell_state = encoder_state
        cell_output = array_ops.zeros(
            [num_decoder_symbols], dtype=dtypes.float32)

        context_state = tensor_array_ops.TensorArray(
            #dtype=dtype, tensor_array_name="greedy_path", size=maximum_length + 1, infer_shape=False)
            dtype=dtype, tensor_array_name="greedy_attention_path", size=0, dynamic_size=True, infer_shape=False)

        # init attention
        attention = init_attention(encoder_state)
      else:
        # construct attention
        #NOTICE since dynamic.. below attention_construct_fn will be called building graph for only once
        attention, scores, alignments = attention_construct_fn(cell_output, attention_keys,
                                           attention_values)

        #attention, scores, alignments = attention_construct_fn(cell_state.h, attention_keys,
        #                                   attention_values)

        if input_text is None:
          cell_output = attention
          cell_output = output_fn(cell_output)  # logits
          next_input_id = math_ops.cast(math_ops.argmax(cell_output, 1), dtype=dtype)
        else:
          #encoder_info = nest.flatten(encoder_state)[0]
          #vocab_size = num_decoder_symbols
          ##ref =  tf.Variable(array_ops.zeros([batch_size, vocab_size], dtype=dtypes.float32))
          ##https://github.com/tensorflow/tensorflow/issues/8604
          #with tf.control_dependencies(None):
          #  #TODO... must fix batch_size right now
          #  cell_output =  tf.Variable(array_ops.zeros([256, vocab_size], dtype=dtypes.float32))
          #indices = melt.batch_values_to_indices(tf.to_int32(input_text))
          #updates = scores
          #tf.scatter_nd_add(cell_output, indices, updates)
          ##print(cell_output, vocab_size)
          ##cell_output = attention
          ##cell_output = array_ops.zeros([batch_size, vocab_size], dtype=dtypes.float32)
          ##cell_output = scores
          #cell_output = tf.convert_to_tensor(cell_output)
          #cell_output = array_ops.zeros(
          #  [256, num_decoder_symbols], dtype=dtypes.float32)
          
          cell_output = attention
          cell_output = output_fn(cell_output)  # logit
          #next_input_id = math_ops.cast(math_ops.argmax(cell_output, 1), dtype=dtype)
          next_input_id = melt.gather2d(input_text, math_ops.cast(math_ops.argmax(scores, 1), dtype=dtype))
          next_input_id = tf.to_int32(next_input_id)
        
        done = math_ops.equal(next_input_id, end_of_sequence_id)

        context_state = context_state.write(time - 1, next_input_id)

      # combine cell_input and attention
      cell_input = array_ops.gather(embeddings, next_input_id) if next_input_id is not None else first_input
      next_input = array_ops.concat([cell_input, attention], 1)
      #next_input = cell_input

      # if time == maxlen, return all true vector
      done = control_flow_ops.cond(
          math_ops.equal(time, maximum_length),
          lambda: array_ops.ones([batch_size,], dtype=dtypes.bool),
          lambda: done)

      return (done, cell_state, next_input, cell_output, context_state)

  return decoder_fn
import tensorflow as tf
## Helper functions ##
def prepare_attention(attention_states,
                      attention_option,
                      num_units,
                      reuse=False):
  """Prepare keys/values/functions for attention.

  Args:
    attention_states: hidden states to attend over.
    attention_option: how to compute attention, either "luong" or "bahdanau".
    num_units: hidden state dimension.
    reuse: whether to reuse variable scope.

  Returns:
    attention_keys: to be compared with target states.
    attention_values: to be used to construct context vectors.
    attention_score_fn: to compute similarity between key and target states.
    attention_construct_fn: to build attention states.
  """

  # Prepare attention keys / values from attention_states
  with variable_scope.variable_scope("attention_keys", reuse=reuse) as scope:
    attention_keys = layers.linear(
        attention_states, num_units, biases_initializer=None, scope=scope)
    #well here do not need ad scope reuse since you get attention_keys and use it  not call this linear again, 
    #attention_construct_fn is different, since you call that function each step during loop 

  attention_values = attention_states

  # Attention score function
  attention_score_fn = _create_attention_score_fn("attention_score", num_units,
                                                  attention_option, reuse)

  # Attention construction function
  attention_construct_fn = _create_attention_construct_fn("attention_construct",
                                                          num_units,
                                                          attention_score_fn,
                                                          reuse)

  return (attention_keys, attention_values, attention_score_fn,
          attention_construct_fn)

def init_attention(encoder_state):
  """Initialize attention. Handling both LSTM and GRU.

  Args:
    encoder_state: The encoded state to initialize the `dynamic_rnn_decoder`.

  Returns:
    attn: initial zero attention vector.
  """

  # Multi- vs single-layer
  # TODO(thangluong): is this the best way to check?
  if isinstance(encoder_state, tuple):
    top_state = encoder_state[-1]
  else:
    top_state = encoder_state

  # LSTM vs GRU
  if isinstance(top_state, rnn_cell_impl.LSTMStateTuple):
    attn = array_ops.zeros_like(top_state.h)
  else:
    attn = array_ops.zeros_like(top_state)

  return attn


def _create_attention_construct_fn(name, num_units, attention_score_fn, reuse):
  """Function to compute attention vectors.

  Args:
    name: to label variables.
    num_units: hidden state dimension.
    attention_score_fn: to compute similarity between key and target states.
    reuse: whether to reuse variable scope.

  Returns:
    attention_construct_fn: to build attention states.
  """
  with variable_scope.variable_scope(name, reuse=reuse) as scope:
    def construct_fn(attention_query, attention_keys, attention_values):
      context, scores, alignments = attention_score_fn(attention_query, attention_keys,
                                   attention_values)
      concat_input = array_ops.concat([attention_query, context], 1)
      #NOTICE! here pass scope which is outside construct_fn func! so always not affected by using env scope, be like
      #seq2seq/main/decode/attention_construct/weights in rnn_decoder.py not seq2seq/main/decode/rnn/loop_function/weights in beam_decoder.py
      attention = layers.linear(
          concat_input, num_units, biases_initializer=None, scope=scope)
      #add this to make it safe using in loop, if only used in dynamic decode do not need this since only call above linear once
      #this kind of witting then add reuse seems safe and elegant 
      #http://stackoverflow.com/questions/38545362/tensorflow-variable-scope-reuse-if-variable-exists
      scope.reuse_variables()
      return attention, scores, alignments

    return construct_fn


# keys: [batch_size, attention_length, attn_size]
# query: [batch_size, 1, attn_size]
# return weights [batch_size, attention_length]
#@function.Defun(func_name="attn_add_fun", noinline=True)
def _attn_add_fun(v, keys, query):
  return math_ops.reduce_sum(v * math_ops.tanh(keys + query), [2])


#TODO: with below will cause load meta graph fail, see textsum/predict.py
#@function.Defun(func_name="attn_mul_fun", noinline=True)
def _attn_mul_fun(keys, query):
  return math_ops.reduce_sum(keys * query, [2])


def _create_attention_score_fn(name,
                               num_units,
                               attention_option,
                               reuse,
                               dtype=dtypes.float32):
  """Different ways to compute attention scores.

  Args:
    name: to label variables.
    num_units: hidden state dimension.
    attention_option: how to compute attention, either "luong" or "bahdanau".
      "bahdanau": additive (Bahdanau et al., ICLR'2015)
      "luong": multiplicative (Luong et al., EMNLP'2015)
    reuse: whether to reuse variable scope.
    dtype: (default: `dtypes.float32`) data type to use.

  Returns:
    attention_score_fn: to compute similarity between key and target states.
  """
  with variable_scope.variable_scope(name, reuse=reuse) as scope:
    if attention_option == "bahdanau":
      query_w = variable_scope.get_variable(
          "attnW", [num_units, num_units], dtype=dtype)
      score_v = variable_scope.get_variable("attnV", [num_units], dtype=dtype)
      scope.reuse_variables()

    def attention_score_fn(query, keys, values):
      """Put attention masks on attention_values using attention_keys and query.

      Args:
        query: A Tensor of shape [batch_size, num_units].
        keys: A Tensor of shape [batch_size, attention_length, num_units].
        values: A Tensor of shape [batch_size, attention_length, num_units].

      Returns:
        context_vector: A Tensor of shape [batch_size, num_units].

      Raises:
        ValueError: if attention_option is neither "luong" or "bahdanau".


      """
      if attention_option == "bahdanau":
        # transform query
        query = math_ops.matmul(query, query_w)

        # reshape query: [batch_size, 1, num_units]
        query = array_ops.reshape(query, [-1, 1, num_units])

        # attn_fun
        scores = _attn_add_fun(score_v, keys, query)
      elif attention_option == "luong":
        # reshape query: [batch_size, 1, num_units]
        query = array_ops.reshape(query, [-1, 1, num_units])

        # attn_fun
        scores = _attn_mul_fun(keys, query)
      else:
        raise ValueError("Unknown attention option %s!" % attention_option)

      # Compute alignment weights
      #   scores: [batch_size, length]
      #   alignments: [batch_size, length]
      # TODO(thangluong): not normalize over padding positions.
      
      tf.add_to_collection('attention_scores', scores)

      alignments = nn_ops.softmax(scores)

      tf.add_to_collection('attention_alignments', alignments)
      #will show 2 times since in outgraph decoding, first time you pass batch_size 1 then batch_size beam_size
      #but in train evaluate mode.. will add many times!
      #print('attention_alignments', tf.get_collection('attention_alignments'))

      # Now calculate the attention-weighted vector.
      alignments = array_ops.expand_dims(alignments, 2)

      context_vector = math_ops.reduce_sum(alignments * values, [1])
      context_vector.set_shape([None, num_units])

      return context_vector, scores, alignments

    return attention_score_fn
