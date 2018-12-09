# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.summary import summary
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as vars_
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer as optimizer_
from tensorflow.python.training import training as train


import tensorflow as tf

try:
  import horovod.keras as hvd
except Exception:
  pass

"""
copy from tensorflow.contrib.layers.python.layers.optimerzers.py version 0.10
"""

"""Optimizer ops for use in layers and tf.learn."""

OPTIMIZER_CLS_NAMES = {
    "Adagrad": train.AdagradOptimizer,
    "Adam": train.AdamOptimizer,
    "Ftrl": train.FtrlOptimizer,
    "Momentum": train.MomentumOptimizer,
    "RMSProp": train.RMSPropOptimizer,
    "SGD": train.GradientDescentOptimizer,
}

OPTIMIZER_SUMMARIES = ["learning_rate",
    "loss",
    "gradients",
    "gradient_norm",]

# from cifar10_multi_gpu_train.py
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

# ## from cifar10_estimator
# def average_gradients(tower_gradvars):
#   import six
#   import itertools
#   # Now compute global loss and gradients.
#   gradvars = []
#   with tf.name_scope('gradient_averaging'):
#     all_grads = {}
#     for grad, var in itertools.chain(*tower_gradvars):
#       if grad is not None:
#         all_grads.setdefault(var, []).append(grad)
#     for var, grads in six.iteritems(all_grads):
#       # Average gradients on the same device as the variables
#       # to which they apply.
#       with tf.device(var.device):
#         if len(grads) == 1:
#           avg_grad = grads[0]
#         else:
#           avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
#       gradvars.append((avg_grad, var))
#   return gradvars

def optimize_loss(losses,
                  global_step,
                  learning_rate,
                  optimizer, 
                  num_gpus=1,
                  gradient_noise_scale=None,
                  gradient_multipliers=None,
                  clip_gradients=None,
                  learning_rate_decay_fn=None,
                  update_ops=None,
                  variables=None,
                  name=None,
                  summaries=["global_gradient_norm"],
                  colocate_gradients_with_ops=False,
                  increment_global_step=True,
                  use_tpu=False,
                  use_horovod=False):
  """Given loss and parameters for optimizer, returns a training op.
  Args:
    loss: Tensor, 0 dimensional.
    global_step: Tensor, step counter for each update.
    learning_rate: float or Tensor, magnitude of update per each training step.
    optimizer: string, class or optimizer instance, used as trainer.
               string should be name of optimizer, like 'SGD',
                 'Adam', 'Adagrad'. Full list in OPTIMIZER_CLS_NAMES constant.
               class should be sub-class of tf.Optimizer that implements
                 `compute_gradients` and `apply_gradients` functions.
               optimizer instance should be instantion of tf.Optimizer sub-class
                 and have `compute_gradients` and `apply_gradients` functions.
    gradient_noise_scale: float or None, adds 0-mean normal noise scaled by this
                          value.
    gradient_multipliers: dict of variables or variable names to floats.
                          If present, gradients for specified
                          variables will be multiplied by given constant.
    clip_gradients: float or `None`, clips gradients by this value.
    moving_average_decay: Deprecated. float or None, takes into account previous
                          loss to make learning smoother due to outliers.
    learning_rate_decay_fn: function, takes `learning_rate` and `global_step`
                            `Tensor`s, returns `Tensor`.
                            Can be used to implement any learning rate decay
                            functions.
                            For example: tf.train.exponential_decay.
    update_ops: list of update `Operation`s to execute at each step. If `None`,
                uses elements of UPDATE_OPS collection.
    variables: list of variables to optimize or
               `None` to use all trainable variables.
    name: The name for this operation is used to scope operations and summaries.
    summaries: List of internal quantities to visualize on tensorboard. If not
               set only the loss and the learning rate will be reported. The
               complete list is in OPTIMIZER_SUMMARIES.
  Returns:
    Training op.
  Raises:
    ValueError: if optimizer is wrong type.
  """
  with vs.variable_scope(name, "OptimizeLoss", losses + [global_step]):
    # Update ops take UPDATE_OPS collection if not provided.
    if update_ops is None:
      update_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS))

    #--from https://github.com/tensorflow/tensorflow/blob/28c3c5dd38e3b397c2cf0acdaa6388dcbf0349f7/tensorflow/contrib/layers/python/layers/optimizers.py
    # Learning rate variable, with possible decay.
    lr = None
    if learning_rate is not None:
      if (isinstance(learning_rate, ops.Tensor) or isinstance(learning_rate, tf.Variable) and
          learning_rate.get_shape().ndims == 0):
        #print('------------------optimize_loss learning rate do nothhing', learning_rate)
        lr = learning_rate
      elif isinstance(learning_rate, float):
        if learning_rate < 0.0:
          raise ValueError("Invalid learning_rate %s.", learning_rate)
        lr = vs.get_variable(
            "learning_rate", [],
            trainable=False,
            initializer=init_ops.constant_initializer(learning_rate))
      else:
        raise ValueError("Learning rate should be 0d Tensor or float. "
                         "Got %s of type %s" % (str(learning_rate),
                                                str(type(learning_rate))))

    if learning_rate is not None and learning_rate_decay_fn is not None:
      if global_step is None:
        raise ValueError("global_step is required for learning_rate_decay_fn.")
      lr = learning_rate_decay_fn(lr, global_step)
        
    # Create optimizer, given specified parameters.
    if isinstance(optimizer, six.string_types):
      if lr is None:
        raise ValueError("Learning rate is None, but should be specified if "
                         "optimizer is string (%s)." % optimizer)
      if optimizer not in OPTIMIZER_CLS_NAMES:
        raise ValueError(
            "Optimizer name should be one of [%s], you provided %s." %
            (", ".join(OPTIMIZER_CLS_NAMES), optimizer))
      opt = OPTIMIZER_CLS_NAMES[optimizer](learning_rate=lr)
    elif (isinstance(optimizer, type) and
          issubclass(optimizer, optimizer_.Optimizer)):
      if lr is None:
        raise ValueError("Learning rate is None, but should be specified if "
                         "optimizer is class (%s)." % optimizer)
      opt = optimizer(learning_rate=lr)
    elif isinstance(optimizer, optimizer_.Optimizer):
      #print('------------------optimize_loss optimizer do nothing', optimizer)
      opt = optimizer
    elif callable(optimizer):
      if learning_rate is not None:
        opt = optimizer(lr)
      else:
        opt = optimizer()
      if not isinstance(opt, optimizer_.Optimizer):
        pass
        # TODO all tf.keras.optimizers
        #raise ValueError("Unrecognized optimizer: function should return "
        #                 "subclass of Optimizer. Got %s." % str(opt))
    else:
      raise ValueError("Unrecognized optimizer: should be string, "
                       "subclass of Optimizer, instance of "
                       "subclass of Optimizer or function with one argument. "
                       "Got %s." % str(optimizer))

    if use_tpu:
      opt = tf.contrib.tpu.CrossShardOptimizer(opt)
    assert not (use_tpu and use_horovod)
    if use_horovod:
      opt = hvd.DistributedOptimizer(opt)

    if num_gpus > 1:
      # Calculate the gradients for each model tower.
      # TODO check below is all ok, right now single gpu using below code will be slower then tf.contrib.optimize_loss  4.5 batch/s -> 3 batch/s
      tower_grads = []
      for i in range(num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % ('tower', i)) as name_scope:
            # All trainable variables, if specific variables are not specified.
            
            #-- TODO trainable_variables affect speed ?
            if variables is None:
             variables = vars_.trainable_variables()
            # Compute gradients.
            loss = losses[i]
            # if update_ops:
            #   loss = control_flow_ops.with_dependencies(list(update_ops), loss)
            #print('------------',)
            try:
              gradients = opt.compute_gradients(loss, 
                                                variables,
                                                colocate_gradients_with_ops=colocate_gradients_with_ops)
            except Exception:
              # try:
              #   gradients = opt.compute_gradients(loss)
              # except Exception:
              gradients = opt.get_updates(loss, params=variables)
            
            #TODO FIXME might have None for example add another predictor to graph 
            #[(None, <tf.Variable 'dual_bow/model_init/emb:0' shape=(29285, 256) dtype=float32_ref>), 
            #(None, <tf.Variable 'dual_bow/main/dual_textsim/encode/text_mlp/linear/weights:0' shape=(256, 256) dtype=float32_ref>),
            #(<tensorflow.python.framework.ops.IndexedSlices object at 0x1b72ff50>, <tf.Variable 'seq2seq/model_init/emb:0' shape=(29285, 256) dtype=float32_ref>)
            #print('-------gradients1', gradients)
            #--now hack use below, TODO why dual_bow.. in introduced when compute gradient of loss as seem not related my seq2seq loss?
            gradients = [x for x in gradients if x[0] is not None]
            # Optionally add gradient noise.
            if gradient_noise_scale is not None:
              gradients = _add_scaled_noise_to_gradients(gradients, gradient_noise_scale)
            # Multiply some gradients.
            if gradient_multipliers is not None:
              gradients = _multiply_gradients(gradients, gradient_multipliers)
            # Optionally clip gradients by global norm.
            if clip_gradients is not None:
              gradients = _clip_gradients_by_norm(gradients, clip_gradients)
            
            #print('-------gradients', gradients)
            tower_grads.append(gradients)
                    
      gradients = average_gradients(tower_grads)
      if "global_gradient_norm" in summaries or "gradient_norm" in summaries:
        summary.scalar("global_norm/gradient_norm",
                    clip_ops.global_norm(list(zip(*gradients))[0]))

      # Add histograms for variables, gradients and gradient norms.
      for gradient, variable in gradients:
        if isinstance(gradient, ops.IndexedSlices):
          grad_values = gradient.values
        else:
          grad_values = gradient

        if grad_values is not None:
          var_name = variable.name.replace(":", "_")
          if "gradients" in summaries:
            summary.histogram("gradients/%s" % var_name, grad_values)
          if "gradient_norm" in summaries:
            summary.scalar("gradient_norm/%s" % var_name,
                          clip_ops.global_norm([grad_values]))

      if clip_gradients is not None and ("global_gradient_norm" in summaries or
                                        "gradient_norm" in summaries):
        summary.scalar("global_norm/clipped_gradient_norm",
                      clip_ops.global_norm(list(zip(*gradients))[0]))
    else:
      loss = losses[0]
      # All trainable variables, if specific variables are not specified.
      if variables is None:
        variables = vars_.trainable_variables()

      # Compute gradients.
      try:
        gradients = opt.compute_gradients(
            loss,
            variables,
            colocate_gradients_with_ops=colocate_gradients_with_ops)
      except Exception:
        # TODO not work for keras
        gradients = opt.get_updates(loss=loss, params=variables)

      # Optionally add gradient noise.
      if gradient_noise_scale is not None:
        gradients = _add_scaled_noise_to_gradients(gradients,
                                                  gradient_noise_scale)

      # Multiply some gradients.
      if gradient_multipliers is not None:
        gradients = _multiply_gradients(gradients, gradient_multipliers)
        if not gradients:
          raise ValueError(
              "Empty list of (gradient, var) pairs encountered. This is most "
              "likely to be caused by an improper value of gradient_multipliers.")

      if "global_gradient_norm" in summaries or "gradient_norm" in summaries:
        summary.scalar("global_norm/gradient_norm",
                      clip_ops.global_norm(list(zip(*gradients))[0]))

      # Optionally clip gradients by global norm.
      if isinstance(clip_gradients, float):
        gradients = _clip_gradients_by_norm(gradients, clip_gradients)
      elif callable(clip_gradients):
        gradients = clip_gradients(gradients)
      elif clip_gradients is not None:
        raise ValueError(
            "Unknown type %s for clip_gradients" % type(clip_gradients))

      # # Add scalar summary for loss.
      # if "loss" in summaries:
      #   summary.scalar("loss", loss)

      # Add histograms for variables, gradients and gradient norms.
      for gradient, variable in gradients:
        if isinstance(gradient, ops.IndexedSlices):
          grad_values = gradient.values
        else:
          grad_values = gradient

        if grad_values is not None:
          var_name = variable.name.replace(":", "_")
          if "gradients" in summaries:
            summary.histogram("gradients/%s" % var_name, grad_values)
          if "gradient_norm" in summaries:
            summary.scalar("gradient_norm/%s" % var_name,
                          clip_ops.global_norm([grad_values]))

      if clip_gradients is not None and ("global_gradient_norm" in summaries or
                                        "gradient_norm" in summaries):
        summary.scalar("global_norm/clipped_gradient_norm",
                      clip_ops.global_norm(list(zip(*gradients))[0]))

    # Create gradient updates.
    grad_updates = opt.apply_gradients(gradients,
                                       global_step=global_step if increment_global_step else None,
                                       name="train")

    # IMPORTANT this is needed for momentum!
    if update_ops:
      grad_updates = [grad_updates]
      #print('--------------------1', grad_updates)
      grad_updates.extend(update_ops)
      #print('-------------------2', update_ops)
      grad_updates = tf.group(*grad_updates)
      #print('-----------grad updates', grad_updates)

    # # Make sure total_loss is valid.
    # final_loss = array_ops.check_numerics(loss, "Loss is inf or nan")

    # # Ensure the train_tensor computes grad_updates.
    # train_tensor = control_flow_ops.with_dependencies([grad_updates], final_loss)

    #return train_tensor
    return grad_updates


def _clip_gradients_by_norm(grads_and_vars, clip_gradients):
  """Clips gradients by global norm."""
  gradients, variables = zip(*grads_and_vars)
  clipped_gradients, _ = clip_ops.clip_by_global_norm(gradients,
                                                      clip_gradients)
  return list(zip(clipped_gradients, variables))


def _add_scaled_noise_to_gradients(grads_and_vars, gradient_noise_scale):
  """Adds scaled noise from a 0-mean normal distribution to gradients."""
  gradients, variables = zip(*grads_and_vars)
  noisy_gradients = []
  for gradient in gradients:
    if gradient is None:
      noisy_gradients.append(None)
      continue
    if isinstance(gradient, ops.IndexedSlices):
      gradient_shape = gradient.dense_shape
    else:
      gradient_shape = gradient.get_shape()
    noise = random_ops.truncated_normal(gradient_shape) * gradient_noise_scale
    noisy_gradients.append(gradient + noise)
  return list(zip(noisy_gradients, variables))


def _multiply_gradients(grads_and_vars, gradient_multipliers):
  """Multiply specified gradients."""
  multiplied_grads_and_vars = []
  for grad, var in grads_and_vars:
    if (grad is not None and (var in gradient_multipliers or var.name in gradient_multipliers)):
      key = var if var in gradient_multipliers else var.name
      multiplier = constant_op.constant(gradient_multipliers[key], dtype=dtypes.float32)
      if isinstance(grad, ops.IndexedSlices):
        grad_values = grad.values * multiplier
        grad = ops.IndexedSlices(grad_values, grad.indices, grad.dense_shape)
      else:
        grad *= multiplier
    multiplied_grads_and_vars.append((grad, var))
  return multiplied_grads_and_vars
