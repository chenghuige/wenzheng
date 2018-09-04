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

"""
depreciated
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import tensor_array_ops

import tensorflow as tf

__all__ = ["beam_decoder_fn_inference"]

def beam_decoder_fn_inference(output_fn, first_input, encoder_state, 
                                embeddings, end_of_sequence_id,
                                maximum_length, num_decoder_symbols, decoder,
                                dtype=dtypes.int32, name=None):
  """ Greedy decoder function for a sequence-to-sequence model used in the
  `dynamic_rnn_decoder`.

  The `greedy_decoder_fn_inference` is a greedy inference function for a
  sequence-to-sequence model. It should be used when `dynamic_rnn_decoder` is
  in the inference mode.

  The `greedy_decoder_fn_inference` is called with a set of the user arguments
  and returns the `decoder_fn`, which can be passed to the
  `dynamic_rnn_decoder`, such that

  ```
  dynamic_fn_inference = greedy_decoder_fn_inference(...)
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
    embeddings: The embeddings matrix used for the decoder sized
    `[num_decoder_symbols, embedding_size]`.
    start_of_sequence_id: The start of sequence ID in the decoder embeddings.
    end_of_sequence_id: The end of sequence ID in the decoder embeddings.
    maximum_length: The maximum allowed of time steps to decode.
    num_decoder_symbols: The number of classes to decode at each time step.
    dtype: (default: `dtypes.int32`) The default data type to use when
    handling integer objects.
    name: (default: `None`) NameScope for the decoder function;
      defaults to "greedy_decoder_fn_inference"

  Returns:
    A decoder function with the required interface of `dynamic_rnn_decoder`
    intended for inference.
  """
  with ops.name_scope(name, "beam_decoder_fn_inference",
                      [output_fn, first_input, encoder_state, embeddings,
                       end_of_sequence_id, maximum_length, num_decoder_symbols, dtype]):
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
    """ Decoder function used in the `dynamic_rnn_decoder` with the purpose of
    inference.

    The main difference between this decoder function and the `decoder_fn` in
    `greedy_decoder_fn_train` is how `next_cell_input` is calculated. In this
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
  """
    with ops.name_scope(name, "greedy_decoder_fn_inference",
                        [time, cell_state, cell_input, cell_output,
                         context_state]):
      if cell_input is not None:
        raise ValueError("Expected cell_input to be None, but saw: %s" %
                         cell_input)
      if cell_output is None:
        # invariant that this is time == 0
        next_input_id = None
        done = array_ops.zeros([batch_size,], dtype=dtypes.bool)
        cell_state = encoder_state
        cell_output = array_ops.zeros([num_decoder_symbols],
                                      dtype=dtypes.float32)
        #context_state = tensor_array_ops.TensorArray(
        ##    #dtype=dtype, tensor_array_name="beam_path", size=1, infer_shape=False)
        #    dtype=dtype, tensor_array_name="beam_path", size=0, dynamic_size=True, infer_shape=False)
        ##context_state = None

        context_state = decoder.finished_beams
      else:
        #cell_output, finished_beams, log_prob_finished_beams = output_fn(cell_output, time)
        cell_output, next_input_id = output_fn(cell_output, time)
        
        done = math_ops.equal(next_input_id, end_of_sequence_id)
        
        #done = tf.zeros_like(next_input_id, dtype=tf.bool)

        #context_state = context_state.write(0, best_path)
        
        #context_state = context_state.write(time - 1, next_input_id)
        
        #context_state = context_state.write(time - 1, best_path)
        
        #context_state = decoder.past_symbols, decoder.finished_beams, decoder.logprobs_finished_beams
        context_state = decoder.finished_beams

      next_input = array_ops.gather(embeddings, next_input_id) if next_input_id is not None else first_input
      # if time == maxlen, return all true vector
      done = control_flow_ops.cond(math_ops.equal(time, maximum_length),
          lambda: array_ops.ones([batch_size,], dtype=dtypes.bool),
          lambda: done)
      return (done, cell_state, next_input, cell_output, context_state)
  return decoder_fn

