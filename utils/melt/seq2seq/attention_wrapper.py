# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""A powerful dynamic attention wrapper object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.contrib.layers.python.layers import layers


__all__ = [
    "AttentionMechanism",
    "AttentionWrapper",
    "AttentionWrapperState",
    "LuongAttention",
    "BahdanauAttention",
    "CoverageBahdanauAttention",
    "CoverageV2BahdanauAttention",
    "hardmax",
    "PointerAttentionWrapperState",
    "PointerAttentionWrapper",
    "ShowTellAttentionWrapperState",
    "ShowTellAttentionWrapper",
    "TwoLayersAttentionWrapperState",
    "TwoLayersAttentionWrapper",
    "CoverageAttentionWrapperState",
    "CoverageAttentionWrapper",
]


_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access


class AttentionMechanism(object):
  pass


def _prepare_memory(memory, memory_sequence_length, check_inner_dims_defined):
  """Convert to tensor and possibly mask `memory`.

  Args:
    memory: `Tensor`, shaped `[batch_size, max_time, ...]`.
    memory_sequence_length: `int32` `Tensor`, shaped `[batch_size]`.
    check_inner_dims_defined: Python boolean.  If `True`, the `memory`
      argument's shape is checked to ensure all but the two outermost
      dimensions are fully defined.

  Returns:
    A (possibly masked), checked, new `memory`.

  Raises:
    ValueError: If `check_inner_dims_defined` is `True` and not
      `memory.shape[2:].is_fully_defined()`.
  """
  memory = nest.map_structure(
      lambda m: ops.convert_to_tensor(m, name="memory"), memory)
  if memory_sequence_length is not None:
    memory_sequence_length = ops.convert_to_tensor(
        memory_sequence_length, name="memory_sequence_length")
  if check_inner_dims_defined:
    def _check_dims(m):
      if not m.get_shape()[2:].is_fully_defined():
        raise ValueError("Expected memory %s to have fully defined inner dims, "
                         "but saw shape: %s" % (m.name, m.get_shape()))
    nest.map_structure(_check_dims, memory)
  if memory_sequence_length is None:
    seq_len_mask = None
  else:
    seq_len_mask = array_ops.sequence_mask(
        memory_sequence_length,
        maxlen=array_ops.shape(nest.flatten(memory)[0])[1],
        dtype=nest.flatten(memory)[0].dtype)
    seq_len_batch_size = (
        memory_sequence_length.shape[0].value
        or array_ops.shape(memory_sequence_length)[0])
  def _maybe_mask(m, seq_len_mask):
    rank = m.get_shape().ndims
    rank = rank if rank is not None else array_ops.rank(m)
    extra_ones = array_ops.ones(rank - 2, dtype=dtypes.int32)
    m_batch_size = m.shape[0].value or array_ops.shape(m)[0]
    if memory_sequence_length is not None:
      message = ("memory_sequence_length and memory tensor batch sizes do not "
                 "match.")
      with ops.control_dependencies([
          check_ops.assert_equal(
              seq_len_batch_size, m_batch_size, message=message)]):
        seq_len_mask = array_ops.reshape(
            seq_len_mask,
            array_ops.concat((array_ops.shape(seq_len_mask), extra_ones), 0))
        return m * seq_len_mask
    else:
      return m
  return nest.map_structure(lambda m: _maybe_mask(m, seq_len_mask), memory)


def _maybe_mask_score(score, memory_sequence_length, score_mask_value):
  if memory_sequence_length is None:
    return score
  message = ("All values in memory_sequence_length must greater than zero.")
  with ops.control_dependencies(
      [check_ops.assert_positive(memory_sequence_length, message=message)]):
    score_mask = array_ops.sequence_mask(
        memory_sequence_length, maxlen=array_ops.shape(score)[1])
    score_mask_values = score_mask_value * array_ops.ones_like(score)
    return array_ops.where(score_mask, score, score_mask_values)


class _BaseAttentionMechanism(AttentionMechanism):
  """A base AttentionMechanism class providing common functionality.

  Common functionality includes:
    1. Storing the query and memory layers.
    2. Preprocessing and storing the memory.
  """

  def __init__(self,
               query_layer,
               memory,
               probability_fn,
               memory_sequence_length=None,
               memory_layer=None,
               check_inner_dims_defined=True,
               score_mask_value=float("-inf"),
               name=None):
    """Construct base AttentionMechanism class.

    Args:
      query_layer: Callable.  Instance of `tf.layers.Layer`.  The layer's depth
        must match the depth of `memory_layer`.  If `query_layer` is not
        provided, the shape of `query` must match that of `memory_layer`.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      probability_fn: A `callable`.  Converts the score and previous alignments
        to probabilities. Its signature should be:
        `probabilities = probability_fn(score, previous_alignments)`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      memory_layer: Instance of `tf.layers.Layer` (may be None).  The layer's
        depth must match the depth of `query_layer`.
        If `memory_layer` is not provided, the shape of `memory` must match
        that of `query_layer`.
      check_inner_dims_defined: Python boolean.  If `True`, the `memory`
        argument's shape is checked to ensure all but the two outermost
        dimensions are fully defined.
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      name: Name to use when creating ops.
    """
    if (query_layer is not None
        and not isinstance(query_layer, layers_base.Layer)):
      raise TypeError(
          "query_layer is not a Layer: %s" % type(query_layer).__name__)
    if (memory_layer is not None
        and not isinstance(memory_layer, layers_base.Layer)):
      raise TypeError(
          "memory_layer is not a Layer: %s" % type(memory_layer).__name__)
    self._query_layer = query_layer
    self._memory_layer = memory_layer
    if not callable(probability_fn):
      raise TypeError("probability_fn must be callable, saw type: %s" %
                      type(probability_fn).__name__)
    self._probability_fn = lambda score, prev: (  # pylint:disable=g-long-lambda
        probability_fn(
            _maybe_mask_score(score, memory_sequence_length, score_mask_value),
            prev))
    with ops.name_scope(
        name, "BaseAttentionMechanismInit", nest.flatten(memory)):
      self._values = _prepare_memory(
          memory, memory_sequence_length,
          check_inner_dims_defined=check_inner_dims_defined)
      self._keys = (
          self.memory_layer(self._values) if self.memory_layer  # pylint: disable=not-callable
          else self._values)
      self._batch_size = (
          self._keys.shape[0].value or array_ops.shape(self._keys)[0])
      self._alignments_size = (self._keys.shape[1].value or
                               array_ops.shape(self._keys)[1])

  @property
  def memory_layer(self):
    return self._memory_layer

  @property
  def query_layer(self):
    return self._query_layer

  @property
  def values(self):
    return self._values

  @property
  def keys(self):
    return self._keys

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def alignments_size(self):
    return self._alignments_size

  def initial_alignments(self, batch_size, dtype):
    """Creates the initial alignment values for the `AttentionWrapper` class.

    This is important for AttentionMechanisms that use the previous alignment
    to calculate the alignment at the next time step (e.g. monotonic attention).

    The default behavior is to return a tensor of all zeros.

    Args:
      batch_size: `int32` scalar, the batch_size.
      dtype: The `dtype`.

    Returns:
      A `dtype` tensor shaped `[batch_size, alignments_size]`
      (`alignments_size` is the values' `max_time`).
    """
    max_time = self._alignments_size
    return _zero_state_tensors(max_time, batch_size, dtype)


class LuongAttention(_BaseAttentionMechanism):
  """Implements Luong-style (multiplicative) attention scoring.

  This attention has two forms.  The first is standard Luong attention,
  as described in:

  Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
  "Effective Approaches to Attention-based Neural Machine Translation."
  EMNLP 2015.  https://arxiv.org/abs/1508.04025

  The second is the scaled form inspired partly by the normalized form of
  Bahdanau attention.

  To enable the second form, construct the object with parameter
  `scale=True`.
  """

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               scale=False,
               probability_fn=None,
               score_mask_value=float("-inf"),
               memory_layer_name='memory_layer',
               name="LuongAttention"):
    """Construct the AttentionMechanism mechanism.

    Args:
      num_units: The depth of the attention mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      scale: Python boolean.  Whether to scale the energy term.
      probability_fn: (optional) A `callable`.  Converts the score to
        probabilities.  The default is @{tf.nn.softmax}. Other options include
        @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
        Its signature should be: `probabilities = probability_fn(score)`.
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      name: Name to use when creating ops.
    """
    # For LuongAttention, we only transform the memory layer; thus
    # num_units **must** match expected the query depth.
    if probability_fn is None:
      probability_fn = nn_ops.softmax
    wrapped_probability_fn = lambda score, _: probability_fn(score)
    super(LuongAttention, self).__init__(
        query_layer=None,
        memory_layer=layers_core.Dense(
            num_units, name=memory_layer_name, _scope=memory_layer_name, use_bias=False),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._scale = scale
    self._name = name

  def __call__(self, query, previous_alignments):
    """Score the query based on the keys and values.

    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.
      previous_alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]`
        (`alignments_size` is memory's `max_time`).

    Returns:
      alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]` (`alignments_size` is memory's
        `max_time`).

    Raises:
      ValueError: If `key` and `query` depths do not match.
    """
    depth = query.get_shape()[-1]
    key_units = self.keys.get_shape()[-1]
    if depth != key_units:
      raise ValueError(
          "Incompatible or unknown inner dimensions between query and keys.  "
          "Query (%s) has units: %s.  Keys (%s) have units: %s.  "
          "Perhaps you need to set num_units to the the keys' dimension (%s)?"
          % (query, depth, self.keys, key_units, key_units))
    dtype = query.dtype

    with variable_scope.variable_scope(None, "luong_attention", [query]):
      ## Reshape from [batch_size, depth] to [batch_size, 1, depth]
      ## for matmul.

      query = array_ops.expand_dims(query, 1)

      ## Inner product along the query units dimension.
      ## matmul shapes: query is [batch_size, 1, depth] and
      ##                keys is [batch_size, max_time, depth].
      ## the inner product is asked to **transpose keys' inner shape** to get a
      ## batched matmul on:
      ##   [batch_size, 1, depth] . [batch_size, depth, max_time]
      ## resulting in an output shape of:
      ##   [batch_time, 1, max_time].
      ## we then squeee out the center singleton dimension.
      
      ##score = math_ops.matmul(query, self.keys, transpose_b=True)
      ##score = array_ops.squeeze(score, [1])
 
      score = math_ops.reduce_sum(self.keys * query, [2])

      if self._scale:
        # Scalar used in weight scaling
        g = variable_scope.get_variable(
            "attention_g", dtype=dtype, initializer=1.)
        score = g * score

    #alignments = self._probability_fn(score, previous_alignments)
    alignments = score
    return alignments


class BahdanauAttention(_BaseAttentionMechanism):
  """Implements Bahdanau-style (additive) attention.

  This attention has two forms.  The first is Bahdanau attention,
  as described in:

  Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
  "Neural Machine Translation by Jointly Learning to Align and Translate."
  ICLR 2015. https://arxiv.org/abs/1409.0473

  The second is the normalized form.  This form is inspired by the
  weight normalization article:

  Tim Salimans, Diederik P. Kingma.
  "Weight Normalization: A Simple Reparameterization to Accelerate
   Training of Deep Neural Networks."
  https://arxiv.org/abs/1602.07868

  To enable the second form, construct the object with parameter
  `normalize=True`.
  """

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               normalize=False,
               probability_fn=None,
               score_mask_value=float("-inf"),
               memory_layer_name='memory_layer',
               name="BahdanauAttention"):
    """Construct the Attention mechanism.

    Args:
      num_units: The depth of the query mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      normalize: Python boolean.  Whether to normalize the energy term.
      probability_fn: (optional) A `callable`.  Converts the score to
        probabilities.  The default is @{tf.nn.softmax}. Other options include
        @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
        Its signature should be: `probabilities = probability_fn(score)`.
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      name: Name to use when creating ops.
    """
    if probability_fn is None:
      probability_fn = nn_ops.softmax
    wrapped_probability_fn = lambda score, _: probability_fn(score)
    super(BahdanauAttention, self).__init__(
        query_layer=layers_core.Dense(
            num_units, name="query_layer", _scope="query_layer", use_bias=False),
        memory_layer=layers_core.Dense(
            num_units, name=memory_layer_name, _scope=memory_layer_name, use_bias=False),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._normalize = normalize
    self._name = name

  def __call__(self, query, previous_alignments):
    """Score the query based on the keys and values.

    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.
      previous_alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]`
        (`alignments_size` is memory's `max_time`).

    Returns:
      alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]` (`alignments_size` is memory's
        `max_time`).
    """
    with variable_scope.variable_scope(None, "bahdanau_attention", [query]):
      processed_query = self.query_layer(query) if self.query_layer else query
      dtype = processed_query.dtype
      # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
      processed_query = array_ops.expand_dims(processed_query, 1)
      keys = self._keys
      v = variable_scope.get_variable(
          "attention_v", [self._num_units], dtype=dtype)
      if self._normalize:
        # Scalar used in weight normalization
        g = variable_scope.get_variable(
            "attention_g", dtype=dtype,
            initializer=math.sqrt((1. / self._num_units)))
        # Bias added prior to the nonlinearity
        b = variable_scope.get_variable(
            "attention_b", [self._num_units], dtype=dtype,
            initializer=init_ops.zeros_initializer())
        # normed_v = g * v / ||v||
        normed_v = g * v * math_ops.rsqrt(
            math_ops.reduce_sum(math_ops.square(v)))
        score = math_ops.reduce_sum(
            normed_v * math_ops.tanh(keys + processed_query + b), [2])
      else:
        score = math_ops.reduce_sum(v * math_ops.tanh(keys + processed_query),
                                    [2])

    #alignments = self._probability_fn(score, previous_alignments)
    alignments = score
    return alignments


class CoverageBahdanauAttention(_BaseAttentionMechanism):
  """Implements Bahdanau-style (additive) attention.

  This attention has two forms.  The first is Bahdanau attention,
  as described in:

  Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
  "Neural Machine Translation by Jointly Learning to Align and Translate."
  ICLR 2015. https://arxiv.org/abs/1409.0473

  The second is the normalized form.  This form is inspired by the
  weight normalization article:

  Tim Salimans, Diederik P. Kingma.
  "Weight Normalization: A Simple Reparameterization to Accelerate
   Training of Deep Neural Networks."
  https://arxiv.org/abs/1602.07868

  To enable the second form, construct the object with parameter
  `normalize=True`.
  """

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               normalize=False,
               probability_fn=None,
               score_mask_value=float("-inf"),
               memory_layer_name='memory_layer',
               name="CoverageBahdanauAttention"):
    """Construct the Attention mechanism.

    Args:
      num_units: The depth of the query mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      normalize: Python boolean.  Whether to normalize the energy term.
      probability_fn: (optional) A `callable`.  Converts the score to
        probabilities.  The default is @{tf.nn.softmax}. Other options include
        @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
        Its signature should be: `probabilities = probability_fn(score)`.
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      name: Name to use when creating ops.
    """
    if probability_fn is None:
      probability_fn = nn_ops.softmax
    wrapped_probability_fn = lambda score, _: probability_fn(score)
    super(CoverageBahdanauAttention, self).__init__(
        query_layer=layers_core.Dense(
            num_units, name="query_layer", _scope="query_layer", use_bias=False),
        memory_layer=layers_core.Dense(
            num_units, name=memory_layer_name, _scope=memory_layer_name, use_bias=False),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._normalize = normalize
    self._name = name

    self._coverage_layer = layers_core.Dense(
            num_units, name="coverage_layer", _scope="coverage_layer", use_bias=False)

  def __call__(self, query, coverage):
    """Score the query based on the keys and values.

    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.
      previous_alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]`
        (`alignments_size` is memory's `max_time`).

    Returns:
      alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]` (`alignments_size` is memory's
        `max_time`).
    """
    with variable_scope.variable_scope(None, "bahdanau_attention", [query]):
      processed_query = self.query_layer(query) if self.query_layer else query
      processed_coverage = self._coverage_layer(coverage)
      dtype = processed_query.dtype
      # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
      processed_query = array_ops.expand_dims(processed_query, 1)
      processed_coverage = array_ops.expand_dims(processed_coverage, 1)
      keys = self._keys
      v = variable_scope.get_variable(
          "attention_v", [self._num_units], dtype=dtype)
      if self._normalize:
        # Scalar used in weight normalization
        g = variable_scope.get_variable(
            "attention_g", dtype=dtype,
            initializer=math.sqrt((1. / self._num_units)))
        # Bias added prior to the nonlinearity
        b = variable_scope.get_variable(
            "attention_b", [self._num_units], dtype=dtype,
            initializer=init_ops.zeros_initializer())
        # normed_v = g * v / ||v||
        normed_v = g * v * math_ops.rsqrt(
            math_ops.reduce_sum(math_ops.square(v)))
        score = math_ops.reduce_sum(
            normed_v * math_ops.tanh(keys + processed_query + processed_coverage + b), [2])
      else:
        score = math_ops.reduce_sum(v * math_ops.tanh(keys + processed_query + processed_coverage),
                                    [2])

    #alignments = self._probability_fn(score, previous_alignments)
    alignments = score
    return alignments

class CoverageV2BahdanauAttention(_BaseAttentionMechanism):
  """Implements Bahdanau-style (additive) attention.

  This attention has two forms.  The first is Bahdanau attention,
  as described in:

  Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
  "Neural Machine Translation by Jointly Learning to Align and Translate."
  ICLR 2015. https://arxiv.org/abs/1409.0473

  The second is the normalized form.  This form is inspired by the
  weight normalization article:

  Tim Salimans, Diederik P. Kingma.
  "Weight Normalization: A Simple Reparameterization to Accelerate
   Training of Deep Neural Networks."
  https://arxiv.org/abs/1602.07868

  To enable the second form, construct the object with parameter
  `normalize=True`.
  """

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               normalize=False,
               probability_fn=None,
               score_mask_value=float("-inf"),
               memory_layer_name='memory_layer',
               name="CoverageV2BahdanauAttention"):
    """Construct the Attention mechanism.

    Args:
      num_units: The depth of the query mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      normalize: Python boolean.  Whether to normalize the energy term.
      probability_fn: (optional) A `callable`.  Converts the score to
        probabilities.  The default is @{tf.nn.softmax}. Other options include
        @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
        Its signature should be: `probabilities = probability_fn(score)`.
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      name: Name to use when creating ops.
    """
    if probability_fn is None:
      probability_fn = nn_ops.softmax
    wrapped_probability_fn = lambda score, _: probability_fn(score)
    super(CoverageV2BahdanauAttention, self).__init__(
        query_layer=layers_core.Dense(
            num_units, name="query_layer", _scope="query_layer", use_bias=False),
        memory_layer=layers_core.Dense(
            num_units, name=memory_layer_name, _scope=memory_layer_name, use_bias=False),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._normalize = normalize
    self._name = name

    self._coverage_layer = layers_core.Dense(
            num_units, name="coverage_layer", _scope="coverage_layer", use_bias=False)

    self._pre_alignment_layer = layers_core.Dense(
            num_units, name="pre_alignment_layer", _scope="pre_alignment_layer", use_bias=False)

    self._input_layer = layers_core.Dense(
            num_units, name="input_layer", _scope="input_layer", use_bias=False)

  def __call__(self, query, coverage, pre_alignment):
    """Score the query based on the keys and values.

    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.
      previous_alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]`
        (`alignments_size` is memory's `max_time`).

    Returns:
      alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]` (`alignments_size` is memory's
        `max_time`).
    """
    #print(query, coverage, pre_alignment, input)
    with variable_scope.variable_scope(None, "bahdanau_attention", [query]):
      processed_query = self.query_layer(query) if self.query_layer else query
      processed_coverage = self._coverage_layer(coverage)
      processed_pre_alignment = self._pre_alignment_layer(pre_alignment)
      dtype = processed_query.dtype
      # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
      processed_query = array_ops.expand_dims(processed_query, 1)
      processed_coverage = array_ops.expand_dims(processed_coverage, 1)
      processed_pre_alignment = array_ops.expand_dims(processed_pre_alignment, 1)
      keys = self._keys
      v = variable_scope.get_variable(
          "attention_v", [self._num_units], dtype=dtype)
      if self._normalize:
        # Scalar used in weight normalization
        g = variable_scope.get_variable(
            "attention_g", dtype=dtype,
            initializer=math.sqrt((1. / self._num_units)))
        # Bias added prior to the nonlinearity
        b = variable_scope.get_variable(
            "attention_b", [self._num_units], dtype=dtype,
            initializer=init_ops.zeros_initializer())
        # normed_v = g * v / ||v||
        normed_v = g * v * math_ops.rsqrt(
            math_ops.reduce_sum(math_ops.square(v)))
        score = math_ops.reduce_sum(
            normed_v * math_ops.tanh(keys + processed_query + processed_coverage + processed_pre_alignment + b), [2])
      else:
        score = math_ops.reduce_sum(v * math_ops.tanh(keys + processed_query + processed_coverage + processed_pre_alignment),
                                    [2])

    #alignments = self._probability_fn(score, previous_alignments)
    alignments = score
    return alignments

class AttentionWrapperState(
    collections.namedtuple("AttentionWrapperState",
                           ("cell_state", "attention", "time", "alignments",
                            "alignment_history"))):
  """`namedtuple` storing the state of a `AttentionWrapper`.

  Contains:

    - `cell_state`: The state of the wrapped `RNNCell` at the previous time
      step.
    - `attention`: The attention emitted at the previous time step.
    - `time`: int32 scalar containing the current time step.
    - `alignments`: The alignment emitted at the previous time step.
    - `alignment_history`: (if enabled) a `TensorArray` containing alignment
       matrices from all time steps.  Call `stack()` to convert to a `Tensor`.
  """

  def clone(self, **kwargs):
    """Clone this object, overriding components provided by kwargs.

    Example:

    ```python
    initial_state = attention_wrapper.zero_state(dtype=..., batch_size=...)
    initial_state = initial_state.clone(cell_state=encoder_state)
    ```

    Args:
      **kwargs: Any properties of the state object to replace in the returned
        `AttentionWrapperState`.

    Returns:
      A new `AttentionWrapperState` whose properties are the same as
      this one, except any overridden properties as provided in `kwargs`.
    """
    return super(AttentionWrapperState, self)._replace(**kwargs)


def hardmax(logits, name=None):
  """Returns batched one-hot vectors.

  The depth index containing the `1` is that of the maximum logit value.

  Args:
    logits: A batch tensor of logit values.
    name: Name to use when creating ops.
  Returns:
    A batched one-hot tensor.
  """
  with ops.name_scope(name, "Hardmax", [logits]):
    logits = ops.convert_to_tensor(logits, name="logits")
    if logits.get_shape()[-1].value is not None:
      depth = logits.get_shape()[-1].value
    else:
      depth = array_ops.shape(logits)[-1]
    return array_ops.one_hot(
        math_ops.argmax(logits, -1), depth, dtype=logits.dtype)

#TODO new tf version will be rnn_cell_impl.RNNCell, now temply modify for tf verison 1.1.0-rc2
#from tensorflow.contrib.rnn import core_rnn_cell
#class AttentionWrapper(core_rnn_cell.RNNCell):
class AttentionWrapper(rnn_cell_impl.RNNCell):
  """Wraps another `RNNCell` with attention.
  """

  def __init__(self,
               cell,
               attention_mechanism,
               attention_layer_size=None,
               alignment_history=False,
               cell_input_fn=None,
               output_attention=True,
               initial_cell_state=None,
               no_context=False,
               output_alignment=False,
               probability_fn=None,
               score_as_alignment=False,
               name=None):
    """Construct the `AttentionWrapper`.

    Args:
      cell: An instance of `RNNCell`.
      attention_mechanism: An instance of `AttentionMechanism`.
      attention_layer_size: Python integer, the depth of the attention (output)
        layer. If None (default), use the context as attention at each time
        step. Otherwise, feed the context and cell output into the attention
        layer to generate attention at each time step.
      alignment_history: Python boolean, whether to store alignment history
        from all time steps in the final output state (currently stored as a
        time major `TensorArray` on which you must call `stack()`).
      cell_input_fn: (optional) A `callable`.  The default is:
        `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
      output_attention: Python bool.  If `True` (default), the output at each
        time step is the attention value.  This is the behavior of Luong-style
        attention mechanisms.  If `False`, the output at each time step is
        the output of `cell`.  This is the beahvior of Bhadanau-style
        attention mechanisms.  In both cases, the `attention` tensor is
        propagated to the next time step via the state and is used there.
        This flag only controls whether the attention mechanism is propagated
        up to the next cell in an RNN stack or to the top RNN output.
      initial_cell_state: The initial state value to use for the cell when
        the user calls `zero_state()`.  Note that if this value is provided
        now, and the user uses a `batch_size` argument of `zero_state` which
        does not match the batch size of `initial_cell_state`, proper
        behavior is not guaranteed.
      no_context: True when use pointer network, when you only want alignments
      name: Name to use when creating ops.
    """
    super(AttentionWrapper, self).__init__(name=name)
    if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
      raise TypeError(
          "cell must be an RNNCell, saw type: %s" % type(cell).__name__)
    if not isinstance(attention_mechanism, AttentionMechanism):
      raise TypeError(
          "attention_mechanism must be a AttentionMechanism, saw type: %s"
          % type(attention_mechanism).__name__)
    if cell_input_fn is None:
      cell_input_fn = (
          lambda inputs, attention: array_ops.concat([inputs, attention], -1))
    else:
      if not callable(cell_input_fn):
        raise TypeError(
            "cell_input_fn must be callable, saw type: %s"
            % type(cell_input_fn).__name__)

    if attention_layer_size is not None:
      self._attention_layer = layers_core.Dense(
          attention_layer_size, name="attention_layer", _scope="attention_layer", use_bias=False)
      self._attention_layer_size = attention_layer_size
    else:
      self._attention_layer = None
      self._attention_layer_size = attention_mechanism.values.get_shape()[
          -1].value

    self._cell = cell
    self._attention_mechanism = attention_mechanism
    self._cell_input_fn = cell_input_fn
    self._output_attention = output_attention
    self._alignment_history = alignment_history
    self._no_context = no_context
    self._output_alignment = output_alignment
    self._probability_fn = nn_ops.softmax if probability_fn is None else probability_fn
    self._score_as_alignment = score_as_alignment
    with ops.name_scope(name, "AttentionWrapperInit"):
      if initial_cell_state is None:
        self._initial_cell_state = None
      else:
        final_state_tensor = nest.flatten(initial_cell_state)[-1]
        state_batch_size = (
            final_state_tensor.shape[0].value
            or array_ops.shape(final_state_tensor)[0])
        error_message = (
            "When constructing AttentionWrapper %s: " % self._base_name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and initial_cell_state.  Are you using "
            "the BeamSearchDecoder?  You may need to tile your initial state "
            "via the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with ops.control_dependencies(
            [check_ops.assert_equal(state_batch_size,
                                    self._attention_mechanism.batch_size,
                                    message=error_message)]):
          self._initial_cell_state = nest.map_structure(
              lambda s: array_ops.identity(s, name="check_initial_cell_state"),
              initial_cell_state)

  @property
  def output_size(self):
    if self._output_alignment:
      return self._attention_mechanism.alignments_size
    if self._output_attention:
      return self._attention_layer_size
    else:
      return self._cell.output_size

  @property
  def state_size(self):
    return AttentionWrapperState(
        cell_state=self._cell.state_size,
        time=tensor_shape.TensorShape([]),
        attention=self._attention_layer_size,
        alignments=self._attention_mechanism.alignments_size,
        alignment_history=())  # alignment_history is sometimes a TensorArray

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      if self._initial_cell_state is not None:
        cell_state = self._initial_cell_state
      else:
        cell_state = self._cell.zero_state(batch_size, dtype)
      error_message = (
          "When calling zero_state of AttentionWrapper %s: " % self._base_name +
          "Non-matching batch sizes between the memory "
          "(encoder output) and the requested batch size.  Are you using "
          "the BeamSearchDecoder?  If so, make sure your encoder output has "
          "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
          "the batch_size= argument passed to zero_state is "
          "batch_size * beam_width.")
      with ops.control_dependencies(
          [check_ops.assert_equal(batch_size,
                                  self._attention_mechanism.batch_size,
                                  message=error_message)]):
        cell_state = nest.map_structure(
            lambda s: array_ops.identity(s, name="checked_cell_state"),
            cell_state)
      if self._alignment_history:
        alignment_history = tensor_array_ops.TensorArray(
            dtype=dtype, size=0, dynamic_size=True)
      else:
        alignment_history = ()
      return AttentionWrapperState(
          cell_state=cell_state,
          time=array_ops.zeros([], dtype=dtypes.int32),
          attention=_zero_state_tensors(self._attention_layer_size, batch_size,
                                        dtype),
          alignments=self._attention_mechanism.initial_alignments(
              batch_size, dtype),
          alignment_history=alignment_history)

  def call(self, inputs, state):
    """Perform a step of attention-wrapped RNN.

    - Step 1: Mix the `inputs` and previous step's `attention` output via
      `cell_input_fn`.
    - Step 2: Call the wrapped `cell` with this input and its previous state.
    - Step 3: Score the cell's output with `attention_mechanism`.
    - Step 4: Calculate the alignments by passing the score through the
      `normalizer`.
    - Step 5: Calculate the context vector as the inner product between the
      alignments and the attention_mechanism's values (memory).
    - Step 6: Calculate the attention output by concatenating the cell output
      and context through the attention layer (a linear layer with
      `attention_layer_size` outputs).

    Args:
      inputs: (Possibly nested tuple of) Tensor, the input at this time step.
      state: An instance of `AttentionWrapperState` containing
        tensors from the previous time step.

    Returns:
      A tuple `(attention_or_cell_output, next_state)`, where:

      - `attention_or_cell_output` depending on `output_attention`.
      - `next_state` is an instance of `AttentionWrapperState`
         containing the state calculated at this time step.
    """
    # Step 1: Calculate the true inputs to the cell based on the
    # previous attention value.
    cell_inputs = self._cell_input_fn(inputs, state.attention)
    cell_state = state.cell_state
    cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

    cell_batch_size = (
        cell_output.shape[0].value or array_ops.shape(cell_output)[0])
    error_message = (
        "When applying AttentionWrapper %s: " % self.name +
        "Non-matching batch sizes between the memory "
        "(encoder output) and the query (decoder output).  Are you using "
        "the BeamSearchDecoder?  You may need to tile your memory input via "
        "the tf.contrib.seq2seq.tile_batch function with argument "
        "multiple=beam_width.")
    # TODO chg hack just remove for outgraph beam search, since attention states will change
    #with ops.control_dependencies(
    #    [check_ops.assert_equal(cell_batch_size,
    #                            self._attention_mechanism.batch_size,
    #                            message=error_message)]):
    cell_output = array_ops.identity(
          cell_output, name="checked_cell_output")

    #alignments = self._attention_mechanism(
    #    cell_output, previous_alignments=state.alignments)

    scores = self._attention_mechanism(
        cell_output, previous_alignments=state.alignments)
    
    alignments = self._probability_fn(scores)
     
    if self._no_context:
      attention = cell_output
    else:
      # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
      
      #expanded_alignments = array_ops.expand_dims(alignments, 1)
      expanded_alignments = array_ops.expand_dims(alignments, 2)
      
      # Context is the inner product of alignments and values along the
      # memory time dimension.
      # alignments shape is
      #   [batch_size, 1, memory_time]
      # attention_mechanism.values shape is
      #   [batch_size, memory_time, attention_mechanism.num_units]
      # the batched matmul is over memory_time, so the output shape is
      #   [batch_size, 1, attention_mechanism.num_units].
      # we then squeeze out the singleton dim.
      attention_mechanism_values = self._attention_mechanism.values
      
      #context = math_ops.matmul(expanded_alignments, attention_mechanism_values)
      #context = array_ops.squeeze(context, [1])

      context = math_ops.reduce_sum(expanded_alignments * attention_mechanism_values, [1])

      if self._attention_layer is not None:
        attention = self._attention_layer(
            array_ops.concat([cell_output, context], 1))
      else:
        attention = context

    if self._alignment_history:
      alignment_history = state.alignment_history.write(
          state.time, alignments)
    else:
      alignment_history = ()

    if self._score_as_alignment:
      alignments = scores

    #TODO maybe outupt both alignments and scores ?
    next_state = AttentionWrapperState(
        time=state.time + 1,
        cell_state=next_cell_state,
        attention=attention,
        alignments=alignments,
        alignment_history=alignment_history)

    #TODO why use output_alignment will be slower for dyanmic_rnn ?
    if self._output_alignment:
      return alignments, next_state

    if self._output_attention:
      return attention, next_state
    else:
      return cell_output, next_state

class PointerAttentionWrapperState(
    collections.namedtuple("PointerAttentionWrapperState",
                           ("cell_state", "attention", "time", "alignments",
                            "alignment_history", "gen_probability"))):
  """`namedtuple` storing the state of a `AttentionWrapper`.

  Contains:

    - `cell_state`: The state of the wrapped `RNNCell` at the previous time
      step.
    - `attention`: The attention emitted at the previous time step.
    - `time`: int32 scalar containing the current time step.
    - `alignments`: The alignment emitted at the previous time step.
    - `alignment_history`: (if enabled) a `TensorArray` containing alignment
       matrices from all time steps.  Call `stack()` to convert to a `Tensor`.
  """

  def clone(self, **kwargs):
    """Clone this object, overriding components provided by kwargs.

    Example:

    ```python
    initial_state = attention_wrapper.zero_state(dtype=..., batch_size=...)
    initial_state = initial_state.clone(cell_state=encoder_state)
    ```

    Args:
      **kwargs: Any properties of the state object to replace in the returned
        `AttentionWrapperState`.

    Returns:
      A new `AttentionWrapperState` whose properties are the same as
      this one, except any overridden properties as provided in `kwargs`.
    """
    return super(PointerAttentionWrapperState, self)._replace(**kwargs)

class PointerAttentionWrapper(rnn_cell_impl.RNNCell):
  """Wraps another `RNNCell` with attention.
  """

  def __init__(self,
               cell,
               attention_mechanism,
               attention_layer_size=None,
               alignment_history=False,
               cell_input_fn=None,
               output_attention=True,
               initial_cell_state=None,
               no_context=False,
               output_alignment=False,
               probability_fn=None,
               score_as_alignment=False,
               mix_pre_alignment=False,
               name=None):
    """Construct the `AttentionWrapper`.

    Args:
      cell: An instance of `RNNCell`.
      attention_mechanism: An instance of `AttentionMechanism`.
      attention_layer_size: Python integer, the depth of the attention (output)
        layer. If None (default), use the context as attention at each time
        step. Otherwise, feed the context and cell output into the attention
        layer to generate attention at each time step.
      alignment_history: Python boolean, whether to store alignment history
        from all time steps in the final output state (currently stored as a
        time major `TensorArray` on which you must call `stack()`).
      cell_input_fn: (optional) A `callable`.  The default is:
        `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
      output_attention: Python bool.  If `True` (default), the output at each
        time step is the attention value.  This is the behavior of Luong-style
        attention mechanisms.  If `False`, the output at each time step is
        the output of `cell`.  This is the beahvior of Bhadanau-style
        attention mechanisms.  In both cases, the `attention` tensor is
        propagated to the next time step via the state and is used there.
        This flag only controls whether the attention mechanism is propagated
        up to the next cell in an RNN stack or to the top RNN output.
      initial_cell_state: The initial state value to use for the cell when
        the user calls `zero_state()`.  Note that if this value is provided
        now, and the user uses a `batch_size` argument of `zero_state` which
        does not match the batch size of `initial_cell_state`, proper
        behavior is not guaranteed.
      no_context: True when use pointer network, when you only want alignments
      name: Name to use when creating ops.
    """
    super(PointerAttentionWrapper, self).__init__(name=name)
    if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
      raise TypeError(
          "cell must be an RNNCell, saw type: %s" % type(cell).__name__)
    if not isinstance(attention_mechanism, AttentionMechanism):
      raise TypeError(
          "attention_mechanism must be a AttentionMechanism, saw type: %s"
          % type(attention_mechanism).__name__)
    if cell_input_fn is None:
      cell_input_fn = (
          lambda inputs, attention: array_ops.concat([inputs, attention], -1))
    else:
      if not callable(cell_input_fn):
        raise TypeError(
            "cell_input_fn must be callable, saw type: %s"
            % type(cell_input_fn).__name__)

    if attention_layer_size is not None:
      self._attention_layer = layers_core.Dense(
          attention_layer_size, name="attention_layer", _scope="attention_layer", use_bias=False)
      self._attention_layer_size = attention_layer_size
    else:
      self._attention_layer = None
      self._attention_layer_size = attention_mechanism.values.get_shape()[
          -1].value

    self._cell = cell
    self._attention_mechanism = attention_mechanism
    self._cell_input_fn = cell_input_fn
    self._output_attention = output_attention
    self._alignment_history = alignment_history
    self._no_context = no_context
    self._output_alignment = output_alignment
    #chg TODO masked softmax support to make inference result stable with different padding length
    #right now use attention will make inference result unstable, but may not hurt inference performance much
    self._probability_fn = nn_ops.softmax if probability_fn is None else probability_fn
    self._score_as_alignment = score_as_alignment
    self._mix_pre_alignment = mix_pre_alignment
    with ops.name_scope(name, "AttentionWrapperInit"):
      if initial_cell_state is None:
        self._initial_cell_state = None
      else:
        final_state_tensor = nest.flatten(initial_cell_state)[-1]
        state_batch_size = (
            final_state_tensor.shape[0].value
            or array_ops.shape(final_state_tensor)[0])
        error_message = (
            "When constructing AttentionWrapper %s: " % self._base_name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and initial_cell_state.  Are you using "
            "the BeamSearchDecoder?  You may need to tile your initial state "
            "via the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with ops.control_dependencies(
            [check_ops.assert_equal(state_batch_size,
                                    self._attention_mechanism.batch_size,
                                    message=error_message)]):
          self._initial_cell_state = nest.map_structure(
              lambda s: array_ops.identity(s, name="check_initial_cell_state"),
              initial_cell_state)

  @property
  def output_size(self):
    if self._output_alignment:
      return self._attention_mechanism.alignments_size
    if self._output_attention:
      return self._attention_layer_size
    else:
      return self._cell.output_size

  @property
  def state_size(self):
    return PointerAttentionWrapperState(
        cell_state=self._cell.state_size,
        time=tensor_shape.TensorShape([]),
        attention=self._attention_layer_size,
        alignments=self._attention_mechanism.alignments_size,
        alignment_history=(), # alignment_history is sometimes a TensorArray
        gen_probability=1.)  

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      if self._initial_cell_state is not None:
        cell_state = self._initial_cell_state
      else:
        cell_state = self._cell.zero_state(batch_size, dtype)
      error_message = (
          "When calling zero_state of AttentionWrapper %s: " % self._base_name +
          "Non-matching batch sizes between the memory "
          "(encoder output) and the requested batch size.  Are you using "
          "the BeamSearchDecoder?  If so, make sure your encoder output has "
          "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
          "the batch_size= argument passed to zero_state is "
          "batch_size * beam_width.")
      with ops.control_dependencies(
          [check_ops.assert_equal(batch_size,
                                  self._attention_mechanism.batch_size,
                                  message=error_message)]):
        cell_state = nest.map_structure(
            lambda s: array_ops.identity(s, name="checked_cell_state"),
            cell_state)
      if self._alignment_history:
        alignment_history = tensor_array_ops.TensorArray(
            dtype=dtype, size=0, dynamic_size=True)
      else:
        alignment_history = ()
      return PointerAttentionWrapperState(
          cell_state=cell_state,
          time=array_ops.zeros([], dtype=dtypes.int32),
          attention=_zero_state_tensors(self._attention_layer_size, batch_size,
                                        dtype),
          alignments=self._attention_mechanism.initial_alignments(
              batch_size, dtype),
          alignment_history=alignment_history,
          gen_probability=array_ops.ones([batch_size, 1], dtype=dtypes.float32))

  def call(self, inputs, state):
    """Perform a step of attention-wrapped RNN.

    - Step 1: Mix the `inputs` and previous step's `attention` output via
      `cell_input_fn`.
    - Step 2: Call the wrapped `cell` with this input and its previous state.
    - Step 3: Score the cell's output with `attention_mechanism`.
    - Step 4: Calculate the alignments by passing the score through the
      `normalizer`.
    - Step 5: Calculate the context vector as the inner product between the
      alignments and the attention_mechanism's values (memory).
    - Step 6: Calculate the attention output by concatenating the cell output
      and context through the attention layer (a linear layer with
      `attention_layer_size` outputs).

    Args:
      inputs: (Possibly nested tuple of) Tensor, the input at this time step.
      state: An instance of `AttentionWrapperState` containing
        tensors from the previous time step.

    Returns:
      A tuple `(attention_or_cell_output, next_state)`, where:

      - `attention_or_cell_output` depending on `output_attention`.
      - `next_state` is an instance of `AttentionWrapperState`
         containing the state calculated at this time step.
    """
    # Step 1: Calculate the true inputs to the cell based on the
    # previous attention value.
    cell_inputs = self._cell_input_fn(inputs, state.attention)
    cell_state = state.cell_state
    cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

    cell_batch_size = (
        cell_output.shape[0].value or array_ops.shape(cell_output)[0])
    error_message = (
        "When applying AttentionWrapper %s: " % self.name +
        "Non-matching batch sizes between the memory "
        "(encoder output) and the query (decoder output).  Are you using "
        "the BeamSearchDecoder?  You may need to tile your memory input via "
        "the tf.contrib.seq2seq.tile_batch function with argument "
        "multiple=beam_width.")
    # TODO chg hack just remove for outgraph beam search, since attention states will change
    #with ops.control_dependencies(
    #    [check_ops.assert_equal(cell_batch_size,
    #                            self._attention_mechanism.batch_size,
    #                            message=error_message)]):
    cell_output = array_ops.identity(
          cell_output, name="checked_cell_output")

    #alignments = self._attention_mechanism(
    #    cell_output, previous_alignments=state.alignments)

    scores = self._attention_mechanism(
        cell_output, previous_alignments=state.alignments)
    
    alignments = self._probability_fn(scores)

    if self._mix_pre_alignment:
      gen_probability_logits = layers.linear(array_ops.concat(
           cell_state, 1), 1, scope='gen_probability')
      ##pointer v2
      # gen_probability_logits = layers.linear(array_ops.concat(
      #      [cell_state[0], cell_state[1], cell_output], 1), 1, scope='gen_probability')
      gen_probability = math_ops.sigmoid(gen_probability_logits)
      alignments = alignments * gen_probability + state.alignments * (1 - gen_probability)
     
    if self._no_context:
      attention = cell_output
    else:
      # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
      
      #expanded_alignments = array_ops.expand_dims(alignments, 1)
      expanded_alignments = array_ops.expand_dims(alignments, 2)
      
      # Context is the inner product of alignments and values along the
      # memory time dimension.
      # alignments shape is
      #   [batch_size, 1, memory_time]
      # attention_mechanism.values shape is
      #   [batch_size, memory_time, attention_mechanism.num_units]
      # the batched matmul is over memory_time, so the output shape is
      #   [batch_size, 1, attention_mechanism.num_units].
      # we then squeeze out the singleton dim.
      attention_mechanism_values = self._attention_mechanism.values
      
      #context = math_ops.matmul(expanded_alignments, attention_mechanism_values)
      #context = array_ops.squeeze(context, [1])

      context = math_ops.reduce_sum(expanded_alignments * attention_mechanism_values, [1])

      if self._attention_layer is not None:
        attention = self._attention_layer(
            array_ops.concat([cell_output, context], 1))
      else:
        attention = context

    if self._alignment_history:
      alignment_history = state.alignment_history.write(
          state.time, alignments)
    else:
      alignment_history = ()

    if self._score_as_alignment:
      alignments = scores

      # # Calculate p_gen
      # if pointer_gen:
      #   with tf.variable_scope('calculate_pgen'):
      #     p_gen = linear([context_vector, state.c, state.h, x], 1, True) # a scalar
      #     p_gen = tf.sigmoid(p_gen)
      #     p_gens.append(p_gen)

    if not self._mix_pre_alignment:
      gen_probability_logits = layers.linear(array_ops.concat(
           [context, cell_output, cell_inputs], 1), 1, scope='gen_probability')
      gen_probability = math_ops.sigmoid(gen_probability_logits)


    #gen_probability = array_ops.ones([array_ops.shape(cell_output)[0], 1], dtype=dtypes.float32)

    #TODO maybe outupt both alignments and scores ?
    next_state = PointerAttentionWrapperState(
        time=state.time + 1,
        cell_state=next_cell_state,
        attention=attention,
        alignments=alignments,
        alignment_history=alignment_history,
        gen_probability=gen_probability)

    #TODO why use output_alignment will be slower for dyanmic_rnn ?
    if self._output_alignment:
      return alignments, next_state

    if self._output_attention:
      return attention, next_state
    else:
      return cell_output, next_state


#TODO move all new wrapper code to another file extended_attention_wrapper
class ShowTellAttentionWrapperState(
    collections.namedtuple("ShowTellAttentionWrapperState",
                          ("cell_state", "attention", "pre_attention", "time", "alignments",
                            "alignment_history", "gen_probability"))):
  """`namedtuple` storing the state of a `AttentionWrapper`.

  Contains:

    - `cell_state`: The state of the wrapped `RNNCell` at the previous time
      step.
    - `attention`: The attention emitted at the previous time step.
    - `time`: int32 scalar containing the current time step.
    - `alignments`: The alignment emitted at the previous time step.
    - `alignment_history`: (if enabled) a `TensorArray` containing alignment
       matrices from all time steps.  Call `stack()` to convert to a `Tensor`.
  """

  def clone(self, **kwargs):
    """Clone this object, overriding components provided by kwargs.

    Example:

    ```python
    initial_state = attention_wrapper.zero_state(dtype=..., batch_size=...)
    initial_state = initial_state.clone(cell_state=encoder_state)
    ```

    Args:
      **kwargs: Any properties of the state object to replace in the returned
        `AttentionWrapperState`.

    Returns:
      A new `AttentionWrapperState` whose properties are the same as
      this one, except any overridden properties as provided in `kwargs`.
    """
    return super(ShowTellAttentionWrapperState, self)._replace(**kwargs)

class ShowTellAttentionWrapper(rnn_cell_impl.RNNCell):
  """Wraps another `RNNCell` with attention.
  """

  def __init__(self,
               cell,
               attention_mechanism,
               attention_layer_size=None,
               alignment_history=False,
               cell_input_fn=None,
               output_attention=True,
               initial_cell_state=None,
               no_context=False,
               output_alignment=False,
               probability_fn=None,
               score_as_alignment=False,
               use_gen_probablity=False,
               name=None):
    """Construct the `AttentionWrapper`.

    Args:
      cell: An instance of `RNNCell`.
      attention_mechanism: An instance of `AttentionMechanism`.
      attention_layer_size: Python integer, the depth of the attention (output)
        layer. If None (default), use the context as attention at each time
        step. Otherwise, feed the context and cell output into the attention
        layer to generate attention at each time step.
      alignment_history: Python boolean, whether to store alignment history
        from all time steps in the final output state (currently stored as a
        time major `TensorArray` on which you must call `stack()`).
      cell_input_fn: (optional) A `callable`.  The default is:
        `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
      output_attention: Python bool.  If `True` (default), the output at each
        time step is the attention value.  This is the behavior of Luong-style
        attention mechanisms.  If `False`, the output at each time step is
        the output of `cell`.  This is the beahvior of Bhadanau-style
        attention mechanisms.  In both cases, the `attention` tensor is
        propagated to the next time step via the state and is used there.
        This flag only controls whether the attention mechanism is propagated
        up to the next cell in an RNN stack or to the top RNN output.
      initial_cell_state: The initial state value to use for the cell when
        the user calls `zero_state()`.  Note that if this value is provided
        now, and the user uses a `batch_size` argument of `zero_state` which
        does not match the batch size of `initial_cell_state`, proper
        behavior is not guaranteed.
      no_context: True when use pointer network, when you only want alignments
      name: Name to use when creating ops.
    """
    super(ShowTellAttentionWrapper, self).__init__(name=name)
    if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
      raise TypeError(
          "cell must be an RNNCell, saw type: %s" % type(cell).__name__)
    if not isinstance(attention_mechanism, AttentionMechanism):
      raise TypeError(
          "attention_mechanism must be a AttentionMechanism, saw type: %s"
          % type(attention_mechanism).__name__)
    if cell_input_fn is None:
      cell_input_fn = (
          lambda inputs, attention: array_ops.concat([inputs, attention], -1))
    else:
      if not callable(cell_input_fn):
        raise TypeError(
            "cell_input_fn must be callable, saw type: %s"
            % type(cell_input_fn).__name__)

    if attention_layer_size is not None:
      self._attention_layer = layers_core.Dense(
          attention_layer_size, name="attention_layer", _scope="attention_layer", use_bias=False)
      self._attention_layer_size = attention_layer_size
    else:
      self._attention_layer = None
      self._attention_layer_size = attention_mechanism.values.get_shape()[
          -1].value

    self._cell = cell
    self._attention_mechanism = attention_mechanism
    self._cell_input_fn = cell_input_fn
    self._output_attention = output_attention
    self._alignment_history = alignment_history
    self._no_context = no_context
    self._output_alignment = output_alignment
    #chg TODO masked softmax support to make inference result stable with different padding length
    #right now use attention will make inference result unstable, but may not hurt inference performance much
    self._probability_fn = nn_ops.softmax if probability_fn is None else probability_fn
    self._score_as_alignment = score_as_alignment
    self._use_gen_probablity = use_gen_probablity
    with ops.name_scope(name, "AttentionWrapperInit"):
      if initial_cell_state is None:
        self._initial_cell_state = None
      else:
        final_state_tensor = nest.flatten(initial_cell_state)[-1]
        state_batch_size = (
            final_state_tensor.shape[0].value
            or array_ops.shape(final_state_tensor)[0])
        error_message = (
            "When constructing AttentionWrapper %s: " % self._base_name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and initial_cell_state.  Are you using "
            "the BeamSearchDecoder?  You may need to tile your initial state "
            "via the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with ops.control_dependencies(
            [check_ops.assert_equal(state_batch_size,
                                    self._attention_mechanism.batch_size,
                                    message=error_message)]):
          self._initial_cell_state = nest.map_structure(
              lambda s: array_ops.identity(s, name="check_initial_cell_state"),
              initial_cell_state)

  @property
  def output_size(self):
    if self._output_alignment:
      return self._attention_mechanism.alignments_size
    if self._output_attention:
      return self._attention_layer_size
    else:
      return self._cell.output_size

  @property
  def state_size(self):
    return ShowTellAttentionWrapperState(
        cell_state=self._cell.state_size,
        time=tensor_shape.TensorShape([]),
        attention=self._attention_layer_size,
        pre_attention=self._attention_layer_size,
        alignments=self._attention_mechanism.alignments_size,
        alignment_history=(), # alignment_history is sometimes a TensorArray
        gen_probability=1)  

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      if self._initial_cell_state is not None:
        cell_state = self._initial_cell_state
      else:
        cell_state = self._cell.zero_state(batch_size, dtype)
      error_message = (
          "When calling zero_state of AttentionWrapper %s: " % self._base_name +
          "Non-matching batch sizes between the memory "
          "(encoder output) and the requested batch size.  Are you using "
          "the BeamSearchDecoder?  If so, make sure your encoder output has "
          "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
          "the batch_size= argument passed to zero_state is "
          "batch_size * beam_width.")
      with ops.control_dependencies(
          [check_ops.assert_equal(batch_size,
                                  self._attention_mechanism.batch_size,
                                  message=error_message)]):
        cell_state = nest.map_structure(
            lambda s: array_ops.identity(s, name="checked_cell_state"),
            cell_state)
      if self._alignment_history:
        alignment_history = tensor_array_ops.TensorArray(
            dtype=dtype, size=0, dynamic_size=True)
      else:
        alignment_history = ()
      return ShowTellAttentionWrapperState(
          cell_state=cell_state,
          time=array_ops.zeros([], dtype=dtypes.int32),
          attention=_zero_state_tensors(self._attention_layer_size, batch_size,
                                        dtype),
          pre_attention=_zero_state_tensors(self._attention_layer_size, batch_size,
                                        dtype),
          alignments=self._attention_mechanism.initial_alignments(
              batch_size, dtype),
          alignment_history=alignment_history,
          gen_probability=array_ops.ones([batch_size, 1], dtype=dtypes.float32))

  def call(self, inputs, state):
    """Perform a step of attention-wrapped RNN.

    - Step 1: Mix the `inputs` and previous step's `attention` output via
      `cell_input_fn`.
    - Step 2: Call the wrapped `cell` with this input and its previous state.
    - Step 3: Score the cell's output with `attention_mechanism`.
    - Step 4: Calculate the alignments by passing the score through the
      `normalizer`.
    - Step 5: Calculate the context vector as the inner product between the
      alignments and the attention_mechanism's values (memory).
    - Step 6: Calculate the attention output by concatenating the cell output
      and context through the attention layer (a linear layer with
      `attention_layer_size` outputs).

    Args:
      inputs: (Possibly nested tuple of) Tensor, the input at this time step.
      state: An instance of `AttentionWrapperState` containing
        tensors from the previous time step.

    Returns:
      A tuple `(attention_or_cell_output, next_state)`, where:

      - `attention_or_cell_output` depending on `output_attention`.
      - `next_state` is an instance of `AttentionWrapperState`
         containing the state calculated at this time step.
    """
    # Step 1: Calculate the true inputs to the cell based on the
    # previous attention value.
    #cell_inputs = self._cell_input_fn(inputs, state.attention)
    #chg add change to use previous two steps attentions 
    cell_inputs = self._cell_input_fn(inputs, array_ops.concat([state.attention, state.pre_attention], -1))

    cell_state = state.cell_state
    cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

    cell_batch_size = (
        cell_output.shape[0].value or array_ops.shape(cell_output)[0])
    error_message = (
        "When applying AttentionWrapper %s: " % self.name +
        "Non-matching batch sizes between the memory "
        "(encoder output) and the query (decoder output).  Are you using "
        "the BeamSearchDecoder?  You may need to tile your memory input via "
        "the tf.contrib.seq2seq.tile_batch function with argument "
        "multiple=beam_width.")
    # TODO chg hack just remove for outgraph beam search, since attention states will change
    #with ops.control_dependencies(
    #    [check_ops.assert_equal(cell_batch_size,
    #                            self._attention_mechanism.batch_size,
    #                            message=error_message)]):
    cell_output = array_ops.identity(
          cell_output, name="checked_cell_output")

    gen_probability_logits = layers.linear(array_ops.concat(
        state.cell_state, 1), 1, scope='gen_probability')
    gen_probability = math_ops.sigmoid(gen_probability_logits)

    if not self._use_gen_probablity:
      gen_probability = state.gen_probability #just 1.

    scores = self._attention_mechanism(
        cell_output, previous_alignments=state.alignments)
    
    alignments = self._probability_fn(scores)
     
    if self._no_context:
      attention = cell_output
    else:
      # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
      
      #expanded_alignments = array_ops.expand_dims(alignments, 1)
      expanded_alignments = array_ops.expand_dims(alignments, 2)
      
      # Context is the inner product of alignments and values along the
      # memory time dimension.
      # alignments shape is
      #   [batch_size, 1, memory_time]
      # attention_mechanism.values shape is
      #   [batch_size, memory_time, attention_mechanism.num_units]
      # the batched matmul is over memory_time, so the output shape is
      #   [batch_size, 1, attention_mechanism.num_units].
      # we then squeeze out the singleton dim.
      attention_mechanism_values = self._attention_mechanism.values
      
      #context = math_ops.matmul(expanded_alignments, attention_mechanism_values)
      #context = array_ops.squeeze(context, [1])

      context = math_ops.reduce_sum(expanded_alignments * attention_mechanism_values, [1])

      context *= gen_probability

      if self._attention_layer is not None:
        attention = self._attention_layer(
            array_ops.concat([cell_output, context], 1))
      else:
        attention = context

    if self._alignment_history:
      alignment_history = state.alignment_history.write(
          state.time, alignments)
    else:
      alignment_history = ()

    if self._score_as_alignment:
      alignments = scores

    #TODO maybe outupt both alignments and scores ?
    next_state = ShowTellAttentionWrapperState(
        time=state.time + 1,
        cell_state=next_cell_state,
        attention=attention,
        pre_attention=state.attention,
        alignments=alignments,
        alignment_history=alignment_history,
        gen_probability=gen_probability)

    #TODO why use output_alignment will be slower for dyanmic_rnn ?
    if self._output_alignment:
      return alignments, next_state

    if self._output_attention:
      return attention, next_state
    else:
      return cell_output, next_state


class TwoLayersAttentionWrapperState(
    collections.namedtuple("TwoLayersAttentionWrapperState",
                           ("cell_state", "attention", "time", 
                            "alignments", "first_layer_alignments",
                            "alignment_history", "gen_probability"))):
  """`namedtuple` storing the state of a `AttentionWrapper`.

  Contains:

    - `cell_state`: The state of the wrapped `RNNCell` at the previous time
      step.
    - `attention`: The attention emitted at the previous time step.
    - `time`: int32 scalar containing the current time step.
    - `alignments`: The alignment emitted at the previous time step.
    - `alignment_history`: (if enabled) a `TensorArray` containing alignment
       matrices from all time steps.  Call `stack()` to convert to a `Tensor`.
  """

  def clone(self, **kwargs):
    """Clone this object, overriding components provided by kwargs.

    Example:

    ```python
    initial_state = attention_wrapper.zero_state(dtype=..., batch_size=...)
    initial_state = initial_state.clone(cell_state=encoder_state)
    ```

    Args:
      **kwargs: Any properties of the state object to replace in the returned
        `AttentionWrapperState`.

    Returns:
      A new `AttentionWrapperState` whose properties are the same as
      this one, except any overridden properties as provided in `kwargs`.
    """
    return super(TwoLayersAttentionWrapperState, self)._replace(**kwargs)

class TwoLayersAttentionWrapper(rnn_cell_impl.RNNCell):
  """Wraps another `RNNCell` with attention.
  """

  def __init__(self,
               cell,
               attention_mechanism,
               attention_mechanism2,
               attention_layer_size=None,
               alignment_history=False,
               cell_input_fn=None,
               output_attention=True,
               initial_cell_state=None,
               no_context=False,
               output_alignment=False,
               probability_fn=None,
               score_as_alignment=False,
               mix_pre_alignment=False,
               name=None):
    """Construct the `AttentionWrapper`.

    Args:
      cell: An instance of `RNNCell`.
      attention_mechanism: An instance of `AttentionMechanism`.
      attention_layer_size: Python integer, the depth of the attention (output)
        layer. If None (default), use the context as attention at each time
        step. Otherwise, feed the context and cell output into the attention
        layer to generate attention at each time step.
      alignment_history: Python boolean, whether to store alignment history
        from all time steps in the final output state (currently stored as a
        time major `TensorArray` on which you must call `stack()`).
      cell_input_fn: (optional) A `callable`.  The default is:
        `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
      output_attention: Python bool.  If `True` (default), the output at each
        time step is the attention value.  This is the behavior of Luong-style
        attention mechanisms.  If `False`, the output at each time step is
        the output of `cell`.  This is the beahvior of Bhadanau-style
        attention mechanisms.  In both cases, the `attention` tensor is
        propagated to the next time step via the state and is used there.
        This flag only controls whether the attention mechanism is propagated
        up to the next cell in an RNN stack or to the top RNN output.
      initial_cell_state: The initial state value to use for the cell when
        the user calls `zero_state()`.  Note that if this value is provided
        now, and the user uses a `batch_size` argument of `zero_state` which
        does not match the batch size of `initial_cell_state`, proper
        behavior is not guaranteed.
      no_context: True when use pointer network, when you only want alignments
      name: Name to use when creating ops.
    """
    super(TwoLayersAttentionWrapper, self).__init__(name=name)
    if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
      raise TypeError(
          "cell must be an RNNCell, saw type: %s" % type(cell).__name__)
    if not isinstance(attention_mechanism, AttentionMechanism):
      raise TypeError(
          "attention_mechanism must be a AttentionMechanism, saw type: %s"
          % type(attention_mechanism).__name__)
    if cell_input_fn is None:
      cell_input_fn = (
          lambda inputs, attention: array_ops.concat([inputs, attention], -1))
    else:
      if not callable(cell_input_fn):
        raise TypeError(
            "cell_input_fn must be callable, saw type: %s"
            % type(cell_input_fn).__name__)

    if attention_layer_size is not None:
      self._attention_layer = layers_core.Dense(
          attention_layer_size, name="attention_layer", _scope="attention_layer", use_bias=False)
      self._attention_layer_size = attention_layer_size
    else:
      self._attention_layer = None
      self._attention_layer_size = attention_mechanism.values.get_shape()[
          -1].value

    self._cell = cell
    self._attention_mechanism = attention_mechanism
    self._attention_mechanism2 = attention_mechanism2
    self._cell_input_fn = cell_input_fn
    self._output_attention = output_attention
    self._alignment_history = alignment_history
    self._no_context = no_context
    self._output_alignment = output_alignment
    self._probability_fn = nn_ops.softmax if probability_fn is None else probability_fn
    self._score_as_alignment = score_as_alignment
    self._mix_pre_alignment = mix_pre_alignment
    with ops.name_scope(name, "AttentionWrapperInit"):
      if initial_cell_state is None:
        self._initial_cell_state = None
      else:
        final_state_tensor = nest.flatten(initial_cell_state)[-1]
        state_batch_size = (
            final_state_tensor.shape[0].value
            or array_ops.shape(final_state_tensor)[0])
        error_message = (
            "When constructing AttentionWrapper %s: " % self._base_name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and initial_cell_state.  Are you using "
            "the BeamSearchDecoder?  You may need to tile your initial state "
            "via the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with ops.control_dependencies(
            [check_ops.assert_equal(state_batch_size,
                                    self._attention_mechanism.batch_size,
                                    message=error_message)]):
          self._initial_cell_state = nest.map_structure(
              lambda s: array_ops.identity(s, name="check_initial_cell_state"),
              initial_cell_state)

  @property
  def output_size(self):
    if self._output_alignment:
      return self._attention_mechanism.alignments_size
    if self._output_attention:
      return self._attention_layer_size
    else:
      return self._cell.output_size

  @property
  def state_size(self):
    gen_probability = 1 if self._mix_pre_alignment else ()
    return TwoLayersAttentionWrapperState(
        cell_state=self._cell.state_size,
        time=tensor_shape.TensorShape([]),
        attention=self._attention_layer_size,
        alignments=self._attention_mechanism2.alignments_size,
        first_layer_alignments=self._attention_mechanism.alignments_size,
        alignment_history=(),
        gen_probability=gen_probability)  # alignment_history is sometimes a TensorArray

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      if self._initial_cell_state is not None:
        cell_state = self._initial_cell_state
      else:
        cell_state = self._cell.zero_state(batch_size, dtype)
      error_message = (
          "When calling zero_state of AttentionWrapper %s: " % self._base_name +
          "Non-matching batch sizes between the memory "
          "(encoder output) and the requested batch size.  Are you using "
          "the BeamSearchDecoder?  If so, make sure your encoder output has "
          "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
          "the batch_size= argument passed to zero_state is "
          "batch_size * beam_width.")
      with ops.control_dependencies(
          [check_ops.assert_equal(batch_size,
                                  self._attention_mechanism.batch_size,
                                  message=error_message)]):
        cell_state = nest.map_structure(
            lambda s: array_ops.identity(s, name="checked_cell_state"),
            cell_state)
      if self._alignment_history:
        alignment_history = tensor_array_ops.TensorArray(
            dtype=dtype, size=0, dynamic_size=True)
      else:
        alignment_history = ()
      if self._mix_pre_alignment:
        gen_probability = array_ops.ones([batch_size, 1], dtype=dtypes.float32)
      else:
        gen_probability = ()

      return TwoLayersAttentionWrapperState(
          cell_state=cell_state,
          time=array_ops.zeros([], dtype=dtypes.int32),
          attention=_zero_state_tensors(self._attention_layer_size, batch_size,
                                        dtype),
          alignments=self._attention_mechanism2.initial_alignments(
              batch_size, dtype),
          first_layer_alignments=self._attention_mechanism.initial_alignments(
              batch_size, dtype),
          alignment_history=alignment_history,
          gen_probability=gen_probability)

  def call(self, inputs, state):
    """Perform a step of attention-wrapped RNN.

    - Step 1: Mix the `inputs` and previous step's `attention` output via
      `cell_input_fn`.
    - Step 2: Call the wrapped `cell` with this input and its previous state.
    - Step 3: Score the cell's output with `attention_mechanism`.
    - Step 4: Calculate the alignments by passing the score through the
      `normalizer`.
    - Step 5: Calculate the context vector as the inner product between the
      alignments and the attention_mechanism's values (memory).
    - Step 6: Calculate the attention output by concatenating the cell output
      and context through the attention layer (a linear layer with
      `attention_layer_size` outputs).

    Args:
      inputs: (Possibly nested tuple of) Tensor, the input at this time step.
      state: An instance of `AttentionWrapperState` containing
        tensors from the previous time step.

    Returns:
      A tuple `(attention_or_cell_output, next_state)`, where:

      - `attention_or_cell_output` depending on `output_attention`.
      - `next_state` is an instance of `AttentionWrapperState`
         containing the state calculated at this time step.
    """
    # Step 1: Calculate the true inputs to the cell based on the
    # previous attention value.
    cell_inputs = self._cell_input_fn(inputs, state.attention)
    cell_state = state.cell_state
    cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

    cell_batch_size = (
        cell_output.shape[0].value or array_ops.shape(cell_output)[0])
    error_message = (
        "When applying AttentionWrapper %s: " % self.name +
        "Non-matching batch sizes between the memory "
        "(encoder output) and the query (decoder output).  Are you using "
        "the BeamSearchDecoder?  You may need to tile your memory input via "
        "the tf.contrib.seq2seq.tile_batch function with argument "
        "multiple=beam_width.")
    # TODO chg hack just remove for outgraph beam search, since attention states will change
    #with ops.control_dependencies(
    #    [check_ops.assert_equal(cell_batch_size,
    #                            self._attention_mechanism.batch_size,
    #                            message=error_message)]):
    cell_output = array_ops.identity(
          cell_output, name="checked_cell_output")


    #alignments = self._attention_mechanism(
    #    cell_output, previous_alignments=state.alignments)

    scores = self._attention_mechanism(
        cell_output, previous_alignments=state.alignments)
    
    alignments = self._probability_fn(scores)

    first_layer_alignments = alignments

    if self._mix_pre_alignment:
      gen_probability_logits = layers.linear(array_ops.concat(
          cell_state, 1), 1, scope='gen_probability')
      # gen_probability_logits = layers.linear(array_ops.concat(
      #      [cell_state[0], cell_state[1], cell_output], 1), 1, scope='gen_probability')
      gen_probability = math_ops.sigmoid(gen_probability_logits)
      alignments = alignments * gen_probability + state.first_layer_alignments * (1 - gen_probability)
     
    if self._no_context:
      attention = cell_output
    else:
      # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
      
      #expanded_alignments = array_ops.expand_dims(alignments, 1)
      expanded_alignments = array_ops.expand_dims(alignments, 2)
      
      # Context is the inner product of alignments and values along the
      # memory time dimension.
      # alignments shape is
      #   [batch_size, 1, memory_time]
      # attention_mechanism.values shape is
      #   [batch_size, memory_time, attention_mechanism.num_units]
      # the batched matmul is over memory_time, so the output shape is
      #   [batch_size, 1, attention_mechanism.num_units].
      # we then squeeze out the singleton dim.
      attention_mechanism_values = self._attention_mechanism.values
      
      #context = math_ops.matmul(expanded_alignments, attention_mechanism_values)
      #context = array_ops.squeeze(context, [1])

      context = math_ops.reduce_sum(expanded_alignments * attention_mechanism_values, [1])

      if self._attention_layer is not None:
        attention = self._attention_layer(
            array_ops.concat([cell_output, context], 1))
      else:
        attention = context

    #----attention layer2
    scores = self._attention_mechanism2(
      attention, previous_alignments=state.alignments)

    alignments = self._probability_fn(scores)

    if self._mix_pre_alignment:
      gen_probability_logits = layers.linear(array_ops.concat(
           cell_state, 1), 1, scope='gen_probability')
      # gen_probability_logits = layers.linear(array_ops.concat(
      #      [cell_state[0], cell_state[1], attention], 1), 1, scope='gen_probability2')
      gen_probability = math_ops.sigmoid(gen_probability_logits)
      alignments = alignments * gen_probability + state.alignments * (1 - gen_probability)
     
    if self._no_context:
      attention = cell_output
    else:
      # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
      
      #expanded_alignments = array_ops.expand_dims(alignments, 1)
      expanded_alignments = array_ops.expand_dims(alignments, 2)
      
      # Context is the inner product of alignments and values along the
      # memory time dimension.
      # alignments shape is
      #   [batch_size, 1, memory_time]
      # attention_mechanism.values shape is
      #   [batch_size, memory_time, attention_mechanism.num_units]
      # the batched matmul is over memory_time, so the output shape is
      #   [batch_size, 1, attention_mechanism.num_units].
      # we then squeeze out the singleton dim.
      attention_mechanism_values = self._attention_mechanism.values
      
      #context = math_ops.matmul(expanded_alignments, attention_mechanism_values)
      #context = array_ops.squeeze(context, [1])

      context = math_ops.reduce_sum(expanded_alignments * attention_mechanism_values, [1])

      if self._attention_layer is not None:
        attention = self._attention_layer(
            array_ops.concat([cell_output, context], 1))
      else:
        attention = context

    if self._alignment_history:
      alignment_history = state.alignment_history.write(
          state.time, alignments)
    else:
      alignment_history = ()

    if self._score_as_alignment:
      alignments = scores

    if not self._mix_pre_alignment:
      # gen_probability_logits = layers.linear(array_ops.concat(
      #      [context, cell_output, cell_inputs], 1), 1, scope='gen_probability')
      # gen_probability = math_ops.sigmoid(gen_probability_logits)
      gen_probability = array_ops.ones([batch_size, 1], dtype=dtypes.float32)

    #TODO maybe outupt both alignments and scores ?
    next_state = TwoLayersAttentionWrapperState(
        time=state.time + 1,
        cell_state=next_cell_state,
        attention=attention,
        alignments=alignments,
        first_layer_alignments=first_layer_alignments,
        alignment_history=alignment_history,
        gen_probability=gen_probability)

    #TODO why use output_alignment will be slower for dyanmic_rnn ?
    if self._output_alignment:
      return alignments, next_state

    if self._output_attention:
      return attention, next_state
    else:
      return cell_output, next_state


class CoverageAttentionWrapperState(
    collections.namedtuple("CoverageAttentionWrapperState",
                           ("cell_state", "attention", "time", "alignments",
                            "alignment_history",  "coverage",))):
  """`namedtuple` storing the state of a `AttentionWrapper`.

  Contains:

    - `cell_state`: The state of the wrapped `RNNCell` at the previous time
      step.
    - `attention`: The attention emitted at the previous time step.
    - `time`: int32 scalar containing the current time step.
    - `alignments`: The alignment emitted at the previous time step.
    - `alignment_history`: (if enabled) a `TensorArray` containing alignment
       matrices from all time steps.  Call `stack()` to convert to a `Tensor`.
  """

  def clone(self, **kwargs):
    """Clone this object, overriding components provided by kwargs.

    Example:

    ```python
    initial_state = attention_wrapper.zero_state(dtype=..., batch_size=...)
    initial_state = initial_state.clone(cell_state=encoder_state)
    ```

    Args:
      **kwargs: Any properties of the state object to replace in the returned
        `AttentionWrapperState`.

    Returns:
      A new `AttentionWrapperState` whose properties are the same as
      this one, except any overridden properties as provided in `kwargs`.
    """
    return super(CoverageAttentionWrapperState, self)._replace(**kwargs)


#TODO new tf version will be rnn_cell_impl.RNNCell, now temply modify for tf verison 1.1.0-rc2
#from tensorflow.contrib.rnn import core_rnn_cell
#class AttentionWrapper(core_rnn_cell.RNNCell):
class CoverageAttentionWrapper(rnn_cell_impl.RNNCell):
  """Wraps another `RNNCell` with attention.
  """

  def __init__(self,
               cell,
               attention_mechanism,
               attention_layer_size=None,
               alignment_history=False,
               cell_input_fn=None,
               output_attention=True,
               initial_cell_state=None,
               no_context=False,
               output_alignment=False,
               probability_fn=None,
               score_as_alignment=False,
               name=None):
    """Construct the `AttentionWrapper`.

    Args:
      cell: An instance of `RNNCell`.
      attention_mechanism: An instance of `AttentionMechanism`.
      attention_layer_size: Python integer, the depth of the attention (output)
        layer. If None (default), use the context as attention at each time
        step. Otherwise, feed the context and cell output into the attention
        layer to generate attention at each time step.
      alignment_history: Python boolean, whether to store alignment history
        from all time steps in the final output state (currently stored as a
        time major `TensorArray` on which you must call `stack()`).
      cell_input_fn: (optional) A `callable`.  The default is:
        `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
      output_attention: Python bool.  If `True` (default), the output at each
        time step is the attention value.  This is the behavior of Luong-style
        attention mechanisms.  If `False`, the output at each time step is
        the output of `cell`.  This is the beahvior of Bhadanau-style
        attention mechanisms.  In both cases, the `attention` tensor is
        propagated to the next time step via the state and is used there.
        This flag only controls whether the attention mechanism is propagated
        up to the next cell in an RNN stack or to the top RNN output.
      initial_cell_state: The initial state value to use for the cell when
        the user calls `zero_state()`.  Note that if this value is provided
        now, and the user uses a `batch_size` argument of `zero_state` which
        does not match the batch size of `initial_cell_state`, proper
        behavior is not guaranteed.
      no_context: True when use pointer network, when you only want alignments
      name: Name to use when creating ops.
    """
    super(CoverageAttentionWrapper, self).__init__(name=name)
    if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
      raise TypeError(
          "cell must be an RNNCell, saw type: %s" % type(cell).__name__)
    if not isinstance(attention_mechanism, AttentionMechanism):
      raise TypeError(
          "attention_mechanism must be a AttentionMechanism, saw type: %s"
          % type(attention_mechanism).__name__)
    if cell_input_fn is None:
      cell_input_fn = (
          lambda inputs, attention: array_ops.concat([inputs, attention], -1))
    else:
      if not callable(cell_input_fn):
        raise TypeError(
            "cell_input_fn must be callable, saw type: %s"
            % type(cell_input_fn).__name__)

    if attention_layer_size is not None:
      self._attention_layer = layers_core.Dense(
          attention_layer_size, name="attention_layer", _scope="attention_layer", use_bias=False)
      self._attention_layer_size = attention_layer_size
    else:
      self._attention_layer = None
      self._attention_layer_size = attention_mechanism.values.get_shape()[
          -1].value

    self._cell = cell
    self._attention_mechanism = attention_mechanism
    self._cell_input_fn = cell_input_fn
    self._output_attention = output_attention
    self._alignment_history = alignment_history
    self._no_context = no_context
    self._output_alignment = output_alignment
    self._probability_fn = nn_ops.softmax if probability_fn is None else probability_fn
    self._score_as_alignment = score_as_alignment
    with ops.name_scope(name, "AttentionWrapperInit"):
      if initial_cell_state is None:
        self._initial_cell_state = None
      else:
        final_state_tensor = nest.flatten(initial_cell_state)[-1]
        state_batch_size = (
            final_state_tensor.shape[0].value
            or array_ops.shape(final_state_tensor)[0])
        error_message = (
            "When constructing AttentionWrapper %s: " % self._base_name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and initial_cell_state.  Are you using "
            "the BeamSearchDecoder?  You may need to tile your initial state "
            "via the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with ops.control_dependencies(
            [check_ops.assert_equal(state_batch_size,
                                    self._attention_mechanism.batch_size,
                                    message=error_message)]):
          self._initial_cell_state = nest.map_structure(
              lambda s: array_ops.identity(s, name="check_initial_cell_state"),
              initial_cell_state)

  @property
  def output_size(self):
    if self._output_alignment:
      return self._attention_mechanism.alignments_size
    if self._output_attention:
      return self._attention_layer_size
    else:
      return self._cell.output_size

  @property
  def state_size(self):
    return CoverageAttentionWrapperState(
        cell_state=self._cell.state_size,
        time=tensor_shape.TensorShape([]),
        attention=self._attention_layer_size,
        alignments=self._attention_mechanism.alignments_size,
        alignment_history=(),
        coverage=self._attention_mechanism.alignments_size)  # alignment_history is sometimes a TensorArray

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      if self._initial_cell_state is not None:
        cell_state = self._initial_cell_state
      else:
        cell_state = self._cell.zero_state(batch_size, dtype)
      error_message = (
          "When calling zero_state of AttentionWrapper %s: " % self._base_name +
          "Non-matching batch sizes between the memory "
          "(encoder output) and the requested batch size.  Are you using "
          "the BeamSearchDecoder?  If so, make sure your encoder output has "
          "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
          "the batch_size= argument passed to zero_state is "
          "batch_size * beam_width.")
      with ops.control_dependencies(
          [check_ops.assert_equal(batch_size,
                                  self._attention_mechanism.batch_size,
                                  message=error_message)]):
        cell_state = nest.map_structure(
            lambda s: array_ops.identity(s, name="checked_cell_state"),
            cell_state)
      if self._alignment_history:
        alignment_history = tensor_array_ops.TensorArray(
            dtype=dtype, size=0, dynamic_size=True)
      else:
        alignment_history = ()
      return CoverageAttentionWrapperState(
          cell_state=cell_state,
          time=array_ops.zeros([], dtype=dtypes.int32),
          attention=_zero_state_tensors(self._attention_layer_size, batch_size,
                                        dtype),
          alignments=self._attention_mechanism.initial_alignments(
              batch_size, dtype),
          alignment_history=alignment_history,
          coverage=self._attention_mechanism.initial_alignments(
              batch_size, dtype))

  def call(self, inputs, state):
    """Perform a step of attention-wrapped RNN.

    - Step 1: Mix the `inputs` and previous step's `attention` output via
      `cell_input_fn`.
    - Step 2: Call the wrapped `cell` with this input and its previous state.
    - Step 3: Score the cell's output with `attention_mechanism`.
    - Step 4: Calculate the alignments by passing the score through the
      `normalizer`.
    - Step 5: Calculate the context vector as the inner product between the
      alignments and the attention_mechanism's values (memory).
    - Step 6: Calculate the attention output by concatenating the cell output
      and context through the attention layer (a linear layer with
      `attention_layer_size` outputs).

    Args:
      inputs: (Possibly nested tuple of) Tensor, the input at this time step.
      state: An instance of `AttentionWrapperState` containing
        tensors from the previous time step.

    Returns:
      A tuple `(attention_or_cell_output, next_state)`, where:

      - `attention_or_cell_output` depending on `output_attention`.
      - `next_state` is an instance of `AttentionWrapperState`
         containing the state calculated at this time step.
    """
    # Step 1: Calculate the true inputs to the cell based on the
    # previous attention value.
    cell_inputs = self._cell_input_fn(inputs, state.attention)
    cell_state = state.cell_state
    cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

    cell_batch_size = (
        cell_output.shape[0].value or array_ops.shape(cell_output)[0])
    error_message = (
        "When applying AttentionWrapper %s: " % self.name +
        "Non-matching batch sizes between the memory "
        "(encoder output) and the query (decoder output).  Are you using "
        "the BeamSearchDecoder?  You may need to tile your memory input via "
        "the tf.contrib.seq2seq.tile_batch function with argument "
        "multiple=beam_width.")
    # TODO chg hack just remove for outgraph beam search, since attention states will change
    #with ops.control_dependencies(
    #    [check_ops.assert_equal(cell_batch_size,
    #                            self._attention_mechanism.batch_size,
    #                            message=error_message)]):
    cell_output = array_ops.identity(
          cell_output, name="checked_cell_output")

    #alignments = self._attention_mechanism(
    #    cell_output, previous_alignments=state.alignments)
    try:
      scores = self._attention_mechanism(
          cell_output, state.coverage)
    except Exception: 
      scores = self._attention_mechanism(
          cell_output, state.coverage, state.alignments)
    
    alignments = self._probability_fn(scores)

    coverage = state.coverage + alignments
     
    if self._no_context:
      attention = cell_output
    else:
      # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
      
      #expanded_alignments = array_ops.expand_dims(alignments, 1)
      expanded_alignments = array_ops.expand_dims(alignments, 2)
      
      # Context is the inner product of alignments and values along the
      # memory time dimension.
      # alignments shape is
      #   [batch_size, 1, memory_time]
      # attention_mechanism.values shape is
      #   [batch_size, memory_time, attention_mechanism.num_units]
      # the batched matmul is over memory_time, so the output shape is
      #   [batch_size, 1, attention_mechanism.num_units].
      # we then squeeze out the singleton dim.
      attention_mechanism_values = self._attention_mechanism.values
      
      #context = math_ops.matmul(expanded_alignments, attention_mechanism_values)
      #context = array_ops.squeeze(context, [1])

      context = math_ops.reduce_sum(expanded_alignments * attention_mechanism_values, [1])

      if self._attention_layer is not None:
        attention = self._attention_layer(
            array_ops.concat([cell_output, context], 1))
      else:
        attention = context

    if self._alignment_history:
      alignment_history = state.alignment_history.write(
          state.time, alignments)
    else:
      alignment_history = ()

    if self._score_as_alignment:
      alignments = scores

    #TODO maybe outupt both alignments and scores ?
    next_state = CoverageAttentionWrapperState(
        time=state.time + 1,
        cell_state=next_cell_state,
        attention=attention,
        alignments=alignments,
        alignment_history=alignment_history,
        coverage=coverage)

    #TODO why use output_alignment will be slower for dyanmic_rnn ?
    if self._output_alignment:
      return alignments, next_state

    if self._output_attention:
      return attention, next_state
    else:
      return cell_output, next_state