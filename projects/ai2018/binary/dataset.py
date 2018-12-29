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
"""CIFAR-10 data set.

See http://www.cs.toronto.edu/~kriz/cifar.html.
"""
import os

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

from collections import namedtuple

import gezi
import melt
logging = melt.logging
from wenzheng.utils import vocabulary

import prepare.config

class Dataset(melt.tfrecords.Dataset):
  def __init__(self, subset='train'):
    super(Dataset, self).__init__(subset)

  def parser(self, example):
    """Parses a single tf.Example into image and label tensors."""
    features_dict = {
      'id':  tf.FixedLenFeature([], tf.string),
      'content_str': tf.FixedLenFeature([], tf.string),
      'content': tf.VarLenFeature(tf.int64),
      'char': tf.VarLenFeature(tf.int64),
      'label': tf.FixedLenFeature([], tf.int64),
      'source':  tf.FixedLenFeature([], tf.string),
      }

    features = tf.parse_single_example(example, features=features_dict)

    content = features['content']
    content = melt.sparse_tensor_to_dense(content)

    features['content'] = content
    label = features['label']

    #if FLAGS.use_char:
    chars = features['char']
    chars = melt.sparse_tensor_to_dense(chars)
    # if FLAGS.char_min_count:
    #   chars = melt.greater_then_set(chars, FLAGS.char_min_count, UNK_ID)
    features['char'] = chars

    x = features
    y = label

    return x, y

