#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   learning_rate_decay.py
#        \author   chenghuige  
#          \date   2017-10-21 06:11:06.077589
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

def exponential_decay(learning_rate, global_step, decay_steps, decay_rate,
                      decay_start_step=0,
                      staircase=False, name=None):
  """Applies exponential decay to the learning rate.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies an exponential decay function
  to a provided initial learning rate.  It requires a `global_step` value to
  compute the decayed learning rate.  You can just pass a TensorFlow variable
  that you increment at each training step.

  The function returns the decayed learning rate.  It is computed as:

  ```python
  decayed_learning_rate = learning_rate *
                          decay_rate ^ (global_step / decay_steps)
  ```

  If the argument `staircase` is `True`, then `global_step / decay_steps` is an
  integer division and the decayed learning rate follows a staircase function.

  Example: decay every 100000 steps with a base of 0.96:

  ```python
  ...
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.1
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                             100000, 0.96, staircase=True)
  # Passing global_step to minimize() will increment it at each step.
  learning_step = (
      tf.train.GradientDescentOptimizer(learning_rate)
      .minimize(...my loss..., global_step=global_step)
  )
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
      Global step to use for the decay computation.  Must not be negative.
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
      Must be positive.  See the decay computation above.
    decay_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The decay rate.
    staircase: Boolean.  If `True` decay the learning rate at discrete intervals
    name: String.  Optional name of the operation.  Defaults to
      'ExponentialDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.

  Raises:
    ValueError: if `global_step` is not supplied.
  """
  if global_step is None:
    raise ValueError("global_step is required for exponential_decay.")
  with ops.name_scope(name, "ExponentialDecay",
                      [learning_rate, global_step,
                       decay_steps, decay_rate,
                       decay_start_step]) as name:
    learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
    dtype = learning_rate.dtype
    global_step = math_ops.cast(global_step, dtype)
    decay_steps = math_ops.cast(decay_steps, dtype)
    decay_rate = math_ops.cast(decay_rate, dtype)
    decay_start_step = math_ops.cast(decay_start_step, dtype)
    p = global_step / decay_steps
    if staircase:
      p = math_ops.floor(p)
    return tf.cond(global_step < decay_start_step,
                   lambda: learning_rate, 
                   lambda: math_ops.multiply(learning_rate, math_ops.pow(decay_rate, p),name=name))

def piecewise_constant(x, boundaries, values, name=None):
  with ops.name_scope(name, "PiecewiseConstant",
                      [x, boundaries, values, name]) as name:
    x = ops.convert_to_tensor(x)  
    #TODO hack chg change this from traing/learning_rate_decay.py
    # Avoid explicit conversion to x's dtype. This could result in faulty
    # comparisons, for example if floats are converted to integers.
    boundaries = ops.convert_n_to_tensor(boundaries, tf.int64)
    for b in boundaries:
      if b.dtype.base_dtype != x.dtype.base_dtype:
        raise ValueError(
            "Boundaries (%s) must have the same dtype as x (%s)." % (
                b.dtype.base_dtype, x.dtype.base_dtype))
    # TODO(rdipietro): Ensure that boundaries' elements are strictly increasing.
    values = ops.convert_n_to_tensor(values)
    for v in values[1:]:
      if v.dtype.base_dtype != values[0].dtype.base_dtype:
        raise ValueError(
            "Values must have elements all with the same dtype (%s vs %s)." % (
                values[0].dtype.base_dtype, v.dtype.base_dtype))

    pred_fn_pairs = {}
    pred_fn_pairs[x <= boundaries[0]] = lambda: values[0]
    pred_fn_pairs[x > boundaries[-1]] = lambda: values[-1]
    for low, high, v in zip(boundaries[:-1], boundaries[1:], values[1:-1]):
      # Need to bind v here; can do this with lambda v=v: ...
      pred = (x > low) & (x <= high)
      pred_fn_pairs[pred] = lambda v=v: v

    # The default isn't needed here because our conditions are mutually
    # exclusive and exhaustive, but tf.case requires it.
    default = lambda: values[0]
    return control_flow_ops.case(pred_fn_pairs, default, exclusive=True) 
