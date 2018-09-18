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

class Dataset(melt.tfrecords.Dataset):
  def __init__(self, subset='train'):
    super(Dataset, self).__init__(subset)

  def parser(self, example):
    """Parses a single tf.Example into image and label tensors."""
    features_dict = {
      'id':  tf.FixedLenFeature([], tf.string),
      'url':  tf.FixedLenFeature([], tf.string),
      'answer': tf.FixedLenFeature([], tf.int64),
      'answer_str':  tf.FixedLenFeature([], tf.string),
      'query': tf.VarLenFeature(tf.int64),
      'query_str':  tf.FixedLenFeature([], tf.string),
      'passage': tf.VarLenFeature(tf.int64),
      'passage_str':  tf.FixedLenFeature([], tf.string),
      'alternatives':  tf.FixedLenFeature([], tf.string),
      'candidates':  tf.FixedLenFeature([], tf.string),
      'type':  tf.FixedLenFeature([], tf.int64),
      }

    features = tf.parse_single_example(example, features=features_dict)

    query = features['query']
    passage = features['passage']
    query = melt.sparse_tensor_to_dense(query)
    passage = melt.sparse_tensor_to_dense(passage)

    if FLAGS.add_start_end:
      query = tf.concat([tf.constant([vocabulary.start_id()], dtype=tf.int64), query, tf.constant([vocabulary.end_id()], dtype=tf.int64)], 0)
    features['query'] = query

    if FLAGS.add_start_end:
      passage = tf.concat([tf.constant([vocabulary.start_id()], dtype=tf.int64), passage, tf.constant([vocabulary.end_id()], dtype=tf.int64)], 0)
    features['passage'] = passage

    if not FLAGS.add_start_end:
      features['content'] = tf.concat([passage, tf.constant([vocabulary.end_id()], dtype=tf.int64), query], 0)
      features['rcontent'] = tf.concat([query, tf.constant([vocabulary.end_id()], dtype=tf.int64), passage], 0)
    else:
        features['content'] = tf.concat([passage, query[1:]], 0)
        features['rcontent'] = tf.concat([query, passage[1:]], 0)    

    answer = features['answer']

    x = features
    y = answer
    return x, y
