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
import algos.config

class Dataset(melt.tfrecords.Dataset):
  def __init__(self, subset='train'):
    super(Dataset, self).__init__(subset)

    # !must use tf.equal not ==, verify this using eager mode! tensor != 1... tf.equal(tensor, 1)
    self.filter_fn = None if not FLAGS.type1_only else lambda x, y: tf.equal(x['type'], 1)
    if FLAGS.type0_only:
      self.filter_fn = lambda x, y: tf.equal(x['type'], 0)
    logging.info('filter_fn', self.filter_fn)

    if FLAGS.type1_count:
      self.count_fn = lambda x, y: tf.cond(tf.equal(x['type'], 1), lambda: FLAGS.type1_count, lambda: 1)
    logging.info('count_fn', self.count_fn)

  def parser(self, example):
    features_dict = {
      'id':  tf.FixedLenFeature([], tf.string),
      'url':  tf.FixedLenFeature([], tf.string),
      'answer': tf.FixedLenFeature([], tf.int64),
      'answer_str':  tf.FixedLenFeature([], tf.string),
      'query': tf.VarLenFeature(tf.int64),
      'query_char': tf.VarLenFeature(tf.int64),
      'query_pos': tf.VarLenFeature(tf.int64),
      'query_str':  tf.FixedLenFeature([], tf.string),
      'passage': tf.VarLenFeature(tf.int64),
      'passage_char': tf.VarLenFeature(tf.int64),
      'passage_pos': tf.VarLenFeature(tf.int64),
      'passage_str':  tf.FixedLenFeature([], tf.string),
      'candidate_neg':  tf.VarLenFeature(tf.int64),
      'candidate_neg_char': tf.VarLenFeature(tf.int64),
      'candidate_neg_pos': tf.VarLenFeature(tf.int64),
      'candidate_pos':  tf.VarLenFeature(tf.int64),
      'candidate_pos_char': tf.VarLenFeature(tf.int64),
      'candidate_pos_pos': tf.VarLenFeature(tf.int64),
      'candidate_na': tf.VarLenFeature(tf.int64),
      'candidate_na_char': tf.VarLenFeature(tf.int64),
      'candidate_na_pos': tf.VarLenFeature(tf.int64),
      'query_char': tf.VarLenFeature(tf.int64),
      'query_pos': tf.VarLenFeature(tf.int64),
      'alternatives':  tf.FixedLenFeature([], tf.string),
      'candidates':  tf.FixedLenFeature([], tf.string),
      'type':  tf.FixedLenFeature([], tf.int64),
      }

    features = tf.parse_single_example(example, features=features_dict)

    query = features['query']
    passage = features['passage']
    candidate_neg = features['candidate_neg']
    candidate_pos = features['candidate_pos']
    candidate_na = features['candidate_na']
    query = melt.sparse_tensor_to_dense(query)
    passage = melt.sparse_tensor_to_dense(passage)
    candidate_neg = melt.sparse_tensor_to_dense(candidate_neg)
    candidate_pos = melt.sparse_tensor_to_dense(candidate_pos)
    candidate_na = melt.sparse_tensor_to_dense(candidate_na)

    def s2d(name):
      x = features[name]
      x = melt.sparse_tensor_to_dense(x)
      features[name] = x

    l = ['query_char', 'query_pos', \
         'passage_char', 'passage_pos', \
         'candidate_neg_char', 'candidate_neg_pos', \
         'candidate_pos_char', 'candidate_pos_pos', \
         'candidate_na_char', 'candidate_na_pos',
        ]
    for name in l:
      s2d(name)

    # def add_start_end(text):
    #   return  tf.concat([tf.constant([vocabulary.start_id()], dtype=tf.int64), text, tf.constant([vocabulary.end_id()], dtype=tf.int64)], 0)

    # if FLAGS.add_start_end:
    #   query = add_start_end(query)
    features['query'] = query

    # if FLAGS.add_start_end:
    #   passage = add_start_end(passage)
    features['passage'] = passage

    # if not FLAGS.add_start_end:
    #   features['content'] = tf.concat([passage, tf.constant([vocabulary.end_id()], dtype=tf.int64), query], 0)
    #   features['rcontent'] = tf.concat([query, tf.constant([vocabulary.end_id()], dtype=tf.int64), passage], 0)
    # else:
    features['content'] = tf.concat([passage, query[1:]], 0)
    features['rcontent'] = tf.concat([query, passage[1:]], 0)    

    # if FLAGS.add_start_end:
    #   candidate_neg = add_start_end(candidate_neg)
    features['candidate_neg'] = candidate_neg

    # if FLAGS.add_start_end:
    #   candidate_pos = add_start_end(candidate_pos)
    features['candidate_pos'] = candidate_pos  

    features['candidate_na'] = candidate_na

    answer = features['answer']

    x = features
    y = answer
    return x, y
