#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   beam_decoder.py
#        \author   chenghuige  
#          \date   2017-01-16 14:15:32.474973
#   \Description  
# ==============================================================================
"""
NOTICE this is ingraph beam decode as opposed to melt.seq2seq.beam_search which is ougraph beam search
but here also not dynamic, just use loop, do all max_steps until stop, for beam search 
actually you can not early stop also since more steps results might be better. 
if length normalizer == 0. then start from same state longer path could not be better.
so you might look at partial result if it not better then all the done beam then mark it as done TODO
but if length normalizer > 0 then.. you can not early stop by theory..
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest

import collections
import melt

def rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None,
                scope=None):
  """RNN decoder for the sequence-to-sequence model.
     copy from tf contrib but modify a bit add state change

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing generated outputs.
      state: The state of each cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
        (Note that in some cases, like basic RNN cell or GRU cell, outputs and
         states can be the same. They are different for LSTM cells though.)
  """
  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs = []
    prev = None
    for i, inp in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp, state = loop_function(i, prev, state)
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      output, state = cell(inp, state)
      outputs.append(output)
      if loop_function is not None:
        prev = output
  return outputs, state


class BeamDecoderOutputs(
    collections.namedtuple("BeamDecoderOutputs",
                           ("paths", "scores", "decoder"))):
  def clone(self, **kwargs):
    return super(BeamDecoderOutputs, self)._replace(**kwargs)

def beam_decode(input, max_words, initial_state, cell, loop_function, scope=None,
                beam_size=7, done_token=0, 
                output_fn=None,  length_normalization_factor=0.,
                prob_as_score=True, topn=1, need_softmax=True,
                fast_greedy=False):
    """
    Beam search decoder
    copy from https://gist.github.com/igormq/000add00702f09029ea4c30eba976e0a
    make small modifications, add more comments and add topn support, and 
    length_normalization_factor

    NOTICE!: not dynamic, for loop based here not tf.while_loop, so ingraph but currently static build graph  
    TODO: consider beam search decoder from https://github.com/google/seq2seq, that implementation might be dynamic!
    but that implementaion only allow batch size 1, so use batch[beam_size] ?

    TODO:make beam_decode dynamic version, by using named_tuple, and tf.nest_map

    NOTICE! fast_greedy mode not test much, just use fast_greed=False, just to calc topn even topn is 1

    Args:
      decoder_inputs: A list of 2D Tensors [batch_size x input_size].
      initial_state: 2D Tensor with shape [batch_size x cell.state_size].
      cell: rnn_cell.RNNCell defining the cell function and size.
      loop_function: This function will be applied to the i-th output
        in order to generate the i+1-st input, and decoder_inputs will be ignored,
        except for the first element ("GO" symbol). 
        Signature -- loop_function(prev_symbol, i) = next
          * prev_symbol is a 1D Tensor of shape [batch_size*beam_size]
          * i is an integer, the step number (when advanced control is needed),
          * next is a 2D Tensor of shape [batch_size*beam_size, input_size].
      scope: Passed to seq2seq.rnn_decoder
      beam_size: An integer beam size to use for each example
      done_token: An integer token that specifies the STOP symbol

      max_words: if max_words == 3, ouputs is as beow, output array len is fixed as 3
      田地 里 </S> 1.87996e-05
      [748  21   9] 3
      [0.10245205285997697, 0.923504919660139, 6.374825504667079e-06]
      一个 弯着腰 </S> 1.59349e-06
      [ 10 415   9] 3
      [0.4947927940753284, 0.04485648322008847, 1.055874923485485e-06] 
      田地 </S> 1.64312e-07
      [748   9   0] 3
      
    Return:
      A tensor of dimensions [batch_size, len(decoder_inputs)] that corresponds to
      the 1-best beam for each batch.
      
    Known limitations:
      * The output sequence consisting of only a STOP symbol is not considered
        (zero-length sequences are not very useful, so this wasn't implemented)
      * The computation graph this creates is messy and not very well-optimized

    TODO: also return path lengths?
    """
    decoder = BeamDecoder(input, 
                          #max_words + 2, #take step start from i==1 and need one more additional done token 
                          max_words,  #now max words inclucde done token  This is safe
                          initial_state, 
                          beam_size=beam_size,
                          done_token=done_token, 
                          output_fn=output_fn,
                          length_normalization_factor=length_normalization_factor,
                          topn=topn,
                          need_softmax=need_softmax,
                          fast_greedy=fast_greedy)
    
    _ = rnn_decoder(
            decoder.decoder_inputs,
            decoder.initial_state,
            cell=cell,
            loop_function = lambda i, prev, state: loop_function(i, prev, state, decoder),
            scope=scope)
    
    if fast_greedy:
      #compat with topn > 1
      score = tf.expand_dims(decoder.logprobs_finished_beams, 1)
      path = tf.expand_dims(decoder.finished_beams, 1)
    else:
      path, score = decoder.calc_topn()
    
    if prob_as_score:
      score = tf.exp(score)

    outputs = BeamDecoderOutputs(
                paths=path,
                scores=score,
                decoder=decoder)

    return outputs, decoder.final_state


#now without beam size 10, about seq->seq 20->4 10w vocab, about 70-100ms, notice the first decode will be slow 400ms+
#exp in deepiu/textsum/inference/infenrence.sh
#TODO add alignments history support for ingraph beam search
class BeamDecoder():
  def __init__(self, input, max_steps, initial_state, beam_size=7, done_token=0,
              batch_size=None, num_classes=None, output_fn=None, 
              length_normalization_factor=0., topn=1, need_softmax=True,
              fast_greedy=False):
    self.length_normalization_factor = length_normalization_factor
    self.topn = topn
    self.need_softmax = need_softmax
    self.beam_size = beam_size
    self.batch_size = batch_size
    if self.batch_size is None:
      self.batch_size = tf.shape(input)[0]
    self.max_len = max_steps
    self.num_classes = num_classes
    self.done_token = done_token
    self.pad_token = 0

    self.output_fn = output_fn
    
    self.past_logprobs = None
    self.past_symbols = None

    self.past_step_logprobs = None

    self.fast_greedy = fast_greedy
    if self.fast_greedy:
      self.finished_beams = tf.zeros((self.batch_size, self.max_len), dtype=tf.int32) 
      self.logprobs_finished_beams = tf.ones((self.batch_size,), dtype=tf.float32) * -float('inf')
    else:
      self.paths_list = []
      self.logprobs_list = []
      self.step_logprobs_list = []

    #for rnn_decoder function need one more step loop, since i== 0 will not output words, take step start from i==1
    self.decoder_inputs = [None] * (self.max_len + 1) 
    self.decoder_inputs[0] = tf.contrib.seq2seq.tile_batch(input, beam_size)

    self.initial_state = initial_state
    self.final_state = None

                      
  def calc_topn(self):
    #[batch_size, beam_size * (max_len-2)]
    logprobs = tf.concat(self.logprobs_list, 1)

    #[batch_size, topn]
    top_logprobs, indices = tf.nn.top_k(logprobs, self.topn)

    #-2 because max_words = max_len - 2  TODO
    length = self.beam_size * (self.max_len - 2)

    indice_offsets = tf.reshape(
      (tf.range(self.batch_size * length) // length) * length,
      [self.batch_size, length])

    indice_offsets = indice_offsets[:, :self.topn]

    #[batch_size, max_len(length index) - 2, beam_size, max_len]
    paths = tf.concat(self.paths_list, 1)
    step_logprobs_path = tf.concat(self.step_logprobs_list, 1)

    #top_paths = paths[:, 0:self.topn, 2, :]
    
    #paths = tf.reshape(paths, [-1, (self.max_len  - 2) * self.beam_size, self.max_len])
    #paths = tf.reshape(paths, [self.batch_size, -1, self.max_len])

    paths = tf.reshape(paths, [-1, self.max_len])
    step_logprobs_path = tf.reshape(step_logprobs_path, [-1, self.max_len])

    top_paths = tf.gather(paths, indice_offsets + indices)
    top_step_logprobs_path = tf.gather(step_logprobs_path, indice_offsets + indices)

    #log probs history
    self.log_probs_history = top_step_logprobs_path

    return top_paths, top_logprobs

  def _get_aligments(self, state):
    attention_size = melt.get_shape(state.alignments, -1)
    alignments = tf.reshape(state.alignments, [-1, self.beam_size, attention_size])
    return alignments
        
  def take_step(self, i, prev, state):
    print('-------------i', i)
    if self.output_fn is not None:
      #[batch_size * beam_size, num_units] -> [batch_size * beam_size, num_classes]
      try:
        output = self.output_fn(prev)
      except Exception:
        output = self.output_fn(prev, state)
    else:
      output = prev

    self.output = output

    #[batch_size * beam_size, num_classes], here use log sofmax
    if self.need_softmax:
      logprobs = tf.nn.log_softmax(output)
    else:
      logprobs = tf.log(tf.maximum(output, 1e-12))
    
    if self.num_classes is None:
      self.num_classes = tf.shape(logprobs)[1]

    #->[batch_size, beam_size, num_classes]
    logprobs_batched = tf.reshape(logprobs,
                                  [-1, self.beam_size, self.num_classes])
    logprobs_batched.set_shape((None, self.beam_size, None))
    
    # Note: masking out entries to -inf plays poorly with top_k, so just subtract out a large number.
    nondone_mask = tf.reshape(
        tf.cast(
          tf.equal(tf.range(self.num_classes), self.done_token),
          tf.float32) * -1e18,
        [1, 1, self.num_classes])

    if self.past_logprobs is None:
      #[batch_size, beam_size, num_classes] -> [batch_size, num_classes]
      #-> past_logprobs[batch_size, beam_size], indices[batch_size, beam_size]
      self.past_logprobs, indices = tf.nn.top_k(
          (logprobs_batched + nondone_mask)[:, 0, :],
          self.beam_size)
      step_logprobs = self.past_logprobs
    else:
      #logprobs_batched [batch_size, beam_size, num_classes] -> [batch_size, beam_size, num_classes]  
      #past_logprobs    [batch_size, beam_size] -> [batch_size, beam_size, 1]
      step_logprobs_batched = logprobs_batched
      logprobs_batched = logprobs_batched + tf.expand_dims(self.past_logprobs, 2)


      #get [batch_size, beam_size] each
      self.past_logprobs, indices = tf.nn.top_k(
          #[batch_size, beam_size * num_classes]
          tf.reshape(logprobs_batched + nondone_mask, 
                     [-1, self.beam_size * self.num_classes]),
          self.beam_size)  

      #get current step logprobs [batch_size, beam_size]
      step_logprobs = tf.gather_nd(tf.reshape(step_logprobs_batched, 
                                              [-1, self.beam_size * self.num_classes]), 
                                   melt.batch_values_to_indices(indices))

    # For continuing to the next symbols [batch_size, beam_size]
    symbols = indices % self.num_classes
    #from wich beam it comes  [batch_size, beam_size]
    parent_refs = indices // self.num_classes

    # NOTE: outputing a zero-length sequence is not supported for simplicity reasons
    #hasky/jupter/tensorflow/beam-search2.ipynb below for mergeing path
    #here when i >= 2
    # tf.reshape(
    #           (tf.range(3 * 5) // 5) * 5,
    #           [3, 5]
    #       ).eval()
    # array([[ 0,  0,  0,  0,  0],
    #        [ 5,  5,  5,  5,  5],
    #        [10, 10, 10, 10, 10]], dtype=int32)
    parent_refs_offsets = tf.reshape(
        (tf.range(self.batch_size * self.beam_size) 
         // self.beam_size) * self.beam_size,
        [self.batch_size, self.beam_size])
    
    #[batch_size, beam_size]
    past_indices = parent_refs + parent_refs_offsets 
    flattened_past_indices = tf.reshape(past_indices, [-1])
    
    if self.past_symbols is None:
      #here when i == 1, when i==0 will not do take step it just do one rnn() get output and use it for i==1 here
      #here will not need to gather state for inital state of each beam is the same
      #[batch_size, beam_size] -> [batch_size, beam_size, 1]
      self.past_symbols = tf.expand_dims(symbols, 2)
      self.past_step_logprobs = tf.expand_dims(step_logprobs, 2)
    else:
      #self.past_symbols [batch_size, beam_size, i - 1] -> past_symbols_batch_major [batch_size * beam_size, i - 1]
      past_symbols_batch_major = tf.reshape(self.past_symbols, [-1, i-1])

      past_step_logprobs_batch_major = tf.reshape(self.past_step_logprobs, [-1, i - 1])
     

      #-> [batch_size, beam_size, i - 1]  
      beam_past_symbols = tf.gather(past_symbols_batch_major,            #[batch_size * beam_size, i - 1]
                                    past_indices                         #[batch_size, beam_size]
                                    )

      beam_past_step_logprobs = tf.gather(past_step_logprobs_batch_major, past_indices)

      if not self.fast_greedy:
        #[batch_size, beam_size, max_len]
        path = tf.concat([self.past_symbols, 
                          tf.ones_like(tf.expand_dims(symbols, 2)) * self.done_token,
                          tf.tile(tf.ones_like(tf.expand_dims(symbols, 2)) * self.pad_token, 
                          [1, 1, self.max_len - i])], 2)

        step_logprobs_path = tf.concat([self.past_step_logprobs, 
                                        tf.expand_dims(step_logprobs_batched[:, :, self.done_token], 2),
                                        tf.tile(tf.ones_like(tf.expand_dims(step_logprobs, 2)) * -float('inf'), 
                                                [1, 1, self.max_len - i])], 2)

        #[batch_size, 1, beam_size, max_len]
        path = tf.expand_dims(path, 1)
        step_logprobs_path = tf.expand_dims(step_logprobs_path, 1)
        self.paths_list.append(path)
        self.step_logprobs_list.append(step_logprobs_path)

      #[batch_size * beam_size, i - 1] -> [batch_size, bam_size, i] the best beam_size paths until step i
      self.past_symbols = tf.concat([beam_past_symbols, tf.expand_dims(symbols, 2)], 2)
      self.past_step_logprobs = tf.concat([beam_past_step_logprobs, tf.expand_dims(step_logprobs, 2)], 2)

      # For finishing the beam 
      #[batch_size, beam_size]
      logprobs_done = logprobs_batched[:, :, self.done_token]
      if not self.fast_greedy:
        self.logprobs_list.append(logprobs_done / i ** self.length_normalization_factor)
      else:
        done_parent_refs = tf.cast(tf.argmax(logprobs_done, 1), tf.int32)
        done_parent_refs_offsets = tf.range(self.batch_size) * self.beam_size

        done_past_symbols = tf.gather(past_symbols_batch_major,
                                      done_parent_refs + done_parent_refs_offsets)

        #[batch_size, max_len]
        symbols_done = tf.concat([done_past_symbols,
                                     tf.ones_like(done_past_symbols[:,0:1]) * self.done_token,
                                     tf.tile(tf.zeros_like(done_past_symbols[:,0:1]),
                                             [1, self.max_len - i])
                                    ], 1)

        #[batch_size, beam_size] -> [batch_size,]
        logprobs_done_max = tf.reduce_max(logprobs_done, 1)
      
        if self.length_normalization_factor > 0:
          logprobs_done_max /= i ** self.length_normalization_factor

        #[batch_size, max_len]
        self.finished_beams = tf.where(logprobs_done_max > self.logprobs_finished_beams,
                                       symbols_done,
                                       self.finished_beams)

        self.logprobs_finished_beams = tf.maximum(logprobs_done_max, self.logprobs_finished_beams)

    
    #TODO not support tf.TensorArray right now, can not use aligment_history in attention_wrapper
    def try_gather(x, indices):
      #if isinstance(x, tf.Tensor) and x.shape.ndims >= 2:
      assert isinstance(x, tf.Tensor)
      if x.shape.ndims >= 2:
        return tf.gather(x, indices)
      else:
        return x

    state = nest.map_structure(lambda x: try_gather(x, flattened_past_indices), state)
    if hasattr(state, 'alignments'):
      attention_size = melt.get_shape(state.alignments, -1)
      alignments = tf.reshape(state.alignments, [-1, self.beam_size, attention_size])
      print('alignments', alignments)


    #->[batch_size * beam_size,]
    symbols_flat = tf.reshape(symbols, [-1])

    self.final_state = state 
    return symbols_flat, state 

