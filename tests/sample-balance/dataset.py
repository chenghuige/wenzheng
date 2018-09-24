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

    ## not work...
    ##self.pos_filter_fn = lambda example: tf.equal(example[0]['type'], 1)
    ## pos and neg balanced sample tested ok
    # self.pos_filter_fn = lambda x, y: tf.equal(x['type'], 1)
    # self.neg_filter_fn = lambda x, y: tf.equal(x['type'], 0)

    # def undersampling_filter(x, y):
    #   prob = tf.cond(tf.equal(x['type'], 1), lambda: 1., lambda: 0.1)
    #   acceptance = tf.less_equal(tf.random_uniform([], dtype=tf.float32), prob)
    #   return acceptance
    # self.filter_fn = undersampling_filter
    
    ## count_fn for over sample tested ok
    self.count_fn = lambda x, y: tf.cond(tf.equal(x['type'], 1), lambda: 10, lambda: 1)

  def parser(self, example):
    features_dict = {
      'id':  tf.FixedLenFeature([], tf.string),
      'url':  tf.FixedLenFeature([], tf.string),
      'answer': tf.FixedLenFeature([], tf.int64),
      'answer_str':  tf.FixedLenFeature([], tf.string),
      'query': tf.VarLenFeature(tf.int64),
      'query_str':  tf.FixedLenFeature([], tf.string),
      'passage': tf.VarLenFeature(tf.int64),
      'passage_str':  tf.FixedLenFeature([], tf.string),
      'candidate_neg':  tf.VarLenFeature(tf.int64),
      'candidate_pos':  tf.VarLenFeature(tf.int64),
      'alternatives':  tf.FixedLenFeature([], tf.string),
      'candidates':  tf.FixedLenFeature([], tf.string),
      'type':  tf.FixedLenFeature([], tf.int64),
      }

    features = tf.parse_single_example(example, features=features_dict)

    query = features['query']
    passage = features['passage']
    candidate_neg = features['candidate_neg']
    candidate_pos = features['candidate_pos']
    query = melt.sparse_tensor_to_dense(query)
    passage = melt.sparse_tensor_to_dense(passage)
    candidate_neg = melt.sparse_tensor_to_dense(candidate_neg)
    candidate_pos = melt.sparse_tensor_to_dense(candidate_pos)

    features['query'] = query
    features['passage'] = passage

    features['content'] = tf.concat([passage, query[1:]], 0)
    features['rcontent'] = tf.concat([query, passage[1:]], 0)    

    features['candidate_neg'] = candidate_neg

    features['candidate_pos'] = candidate_pos  

    answer = features['answer']

    x = features
    y = answer
    return x, y
