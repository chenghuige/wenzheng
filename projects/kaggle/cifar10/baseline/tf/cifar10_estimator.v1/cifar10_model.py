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
"""Model class for Cifar10 Dataset."""
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model_base

import melt


class ResNetCifar10(model_base.ResNet):
  """Cifar10 model with ResNetV1 and basic residual block."""

  def __init__(self,
               num_layers,
               is_training,
               batch_norm_decay,
               batch_norm_epsilon,
               data_format='channels_first'):
    super(ResNetCifar10, self).__init__(
        is_training,
        data_format,
        batch_norm_decay,
        batch_norm_epsilon
    )
    self.n = (num_layers - 2) // 6
    # Add one in case label starts with 1. No impact if label starts with 0.
    self.num_classes = 10 + 1
    self.filters = [16, 16, 32, 64]
    self.strides = [1, 2, 2]
    
  def init_predict(self, input_data_format='channels_last'):
    #self.image_feed = tf.placeholder_with_default(tf.constant([test_image]), [None, ], name='image_feature')
    self.image_feed =  tf.placeholder(tf.string, [None,], name='image')
    tf.add_to_collection('feed', self.image_feed)
    image = tf.map_fn(lambda img: melt.image.decode_image(img, image_format='png', dtype=tf.float32),
                      self.image_feed, dtype=tf.float32)
    self.predict(image)
    tf.add_to_collection('classes', self.pred['classes'])
    tf.add_to_collection('probabilities', self.pred['probabilities'])
    tf.add_to_collection('logits', self.logits)
    tf.add_to_collection('pre_logits', self.pre_logits)

  def forward_pass(self, x, input_data_format='channels_last'):
    # TODO.. without this forward var scope inference_fn will cause problem for self._conv as try to add conv 43.. FIMXE
    with tf.variable_scope('forward'):
      """Build the core model within the graph."""
      if self._data_format != input_data_format:
        if input_data_format == 'channels_last':
          # Computation requires channels_first.
          x = tf.transpose(x, [0, 3, 1, 2])
        else:
          # Computation requires channels_last.
          x = tf.transpose(x, [0, 2, 3, 1])

      # Image standardization.
      x = x / 128 - 1

      x = self._conv(x, 3, 16, 1)
      x = self._batch_norm(x)
      x = self._relu(x)

      # Use basic (non-bottleneck) block and ResNet V1 (post-activation).
      res_func = self._residual_v1

      # 3 stages of block stacking.
      for i in range(3):
        with tf.name_scope('stage'):
          for j in range(self.n):
            if j == 0:
              # First block in a stage, filters and strides may change.
              x = res_func(x, 3, self.filters[i], self.filters[i + 1],
                          self.strides[i])
            else:
              # Following blocks in a stage, constant filters and unit stride.
              x = res_func(x, 3, self.filters[i + 1], self.filters[i + 1], 1)

      x = self._global_avg_pool(x)
      self.pre_logits = x 
      
      x = self._fully_connected(x, self.num_classes)

      self.logits = x

      return x

  def predict(self, x=None, input_data_format='channels_last'):
    if x is not None:
      self.forward_pass(x, input_data_format)
    
    logits = self.logits
    pred = {
      'classes': tf.to_int32(tf.argmax(input=logits, axis=1)),
      'probabilities': tf.nn.softmax(logits)
    }

    self.pred = pred

    return pred

