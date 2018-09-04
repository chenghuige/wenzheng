# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from nets import alexnet
from nets import cifarnet
from nets import inception
from nets import lenet
from nets import mobilenet_v1
from nets import overfeat
from nets import resnet_v1
from nets import resnet_v2
from nets import vgg

slim = tf.contrib.slim

base_networks_map = {
                'inception_v1': inception.inception_v1_base,
                'inception_v2': inception.inception_v2_base,
                'inception_v3': inception.inception_v3_base,
                'inception_v4': inception.inception_v4_base,
                'inception_resnet_v2': inception.inception_resnet_v2_base,
               }

arg_scopes_map = {'alexnet_v2': alexnet.alexnet_v2_arg_scope,
                  'cifarnet': cifarnet.cifarnet_arg_scope,
                  'overfeat': overfeat.overfeat_arg_scope,
                  'vgg_a': vgg.vgg_arg_scope,
                  'vgg_16': vgg.vgg_arg_scope,
                  'vgg_19': vgg.vgg_arg_scope,
                  'inception_v1': inception.inception_v3_arg_scope,
                  'inception_v2': inception.inception_v3_arg_scope,
                  'inception_v3': inception.inception_v3_arg_scope,
                  'inception_v4': inception.inception_v4_arg_scope,
                  'inception_resnet_v2':
                  inception.inception_resnet_v2_arg_scope,
                  'lenet': lenet.lenet_arg_scope,
                  'resnet_v1_50': resnet_v1.resnet_arg_scope,
                  'resnet_v1_101': resnet_v1.resnet_arg_scope,
                  'resnet_v1_152': resnet_v1.resnet_arg_scope,
                  'resnet_v1_200': resnet_v1.resnet_arg_scope,
                  'resnet_v2_50': resnet_v2.resnet_arg_scope,
                  'resnet_v2_101': resnet_v2.resnet_arg_scope,
                  'resnet_v2_152': resnet_v2.resnet_arg_scope,
                  'resnet_v2_200': resnet_v2.resnet_arg_scope,
                  'mobilenet_v1': mobilenet_v1.mobilenet_v1_arg_scope,
                  'mobilenet_v1_075': mobilenet_v1.mobilenet_v1_arg_scope,
                  'mobilenet_v1_050': mobilenet_v1.mobilenet_v1_arg_scope,
                  'mobilenet_v1_025': mobilenet_v1.mobilenet_v1_arg_scope,
                 }


def get_base_network_fn(name, weight_decay=0.0):
  """Returns a network_fn such as `logits, end_points = network_fn(images)`.

  Args:
    name: The name of the network.
    weight_decay: The l2 coefficient for the model weights.

  Returns:
    base_network_fn: A function that applies the model to a batch of images. It has
      the following signature:
        net, end_points = network_fn(images)
  Raises:
    ValueError: If network `name` is not recognized.
  """
  if name not in base_networks_map:
    #raise ValueError('Name of network unknown %s' % name)
    return None
  func = base_networks_map[name]
  @functools.wraps(func)
  def base_network_fn(images):
    arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
      return func(images)
  if hasattr(func, 'default_image_size'):
    base_network_fn.default_image_size = func.default_image_size

  return base_network_fn
