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

import gezi
import melt
logging = melt.logging

class Dataset(object):
  def __init__(self, subset='train'):
    self.subset = subset
    self.filter_fn = None
    self.pos_filter_fn = None
    self.neg_filter_fn = None 
    self.count_fn = None

  def get_filenames(self):
    if self.subset in ['train', 'valid', 'test']:
      if self.subset == 'train':
        return gezi.list_files(FLAGS.train_input)
      elif self.subset == 'valid':
        return gezi.list_files(FLAGS.valid_input)
      elif self.subset == 'test':
        return gezi.list_files(FLAGS.test_input)
    else:
      raise ValueError('Invalid data subset "%s"' % self.subset)

  def parser(self, example):
    pass

  def make_batch(self, 
                 batch_size=None, 
                 filenames=None,
                 initializable=True,
                 repeat=None,
                 return_iterator=True):
    """Read the images and labels from 'filenames'."""
    #with tf.device('/cpu:0'):
    batch_size = batch_size or FLAGS.batch_size
    filenames = filenames or self.get_filenames()
    logging.info(self.subset, 'num files', len(filenames))
    assert filenames, self.subset
    min_queue_examples = 20000
    if repeat is None:
      if tf.executing_eagerly():
        repeat = False 
      else:
        if self.subset == 'train' or melt.num_gpus() > 1:
          repeat = True
        else:
          repeat = False

    if self.subset == 'train':
      shuffle_files=True 
      fix_sequence = False
    else:
      shuffle_files = False
      fix_sequence = True

    balance_pos_neg=False
    if self.pos_filter_fn and self.neg_filter_fn:
      balance_pos_neg = True

    # for bow using cpu 69 insts/s using gpu 54 inst/s
    with tf.device('/cpu:0'):
      return melt.dataset_decode.inputs(
        filenames, 
        decode_fn=self.parser,
        batch_size=batch_size,
        num_threads=FLAGS.num_threads,
        shuffle_files=shuffle_files,
        fix_sequence=fix_sequence,
        buffer_size=min_queue_examples + 3 * batch_size if not FLAGS.buffer_size else FLAGS.buffer_size,
        initializable=initializable,
        repeat=repeat,
        bucket_boundaries=FLAGS.buckets,
        bucket_batch_sizes=FLAGS.batch_sizes,
        length_index=FLAGS.length_index,
        length_key=FLAGS.length_key,
        seed=FLAGS.random_seed,
        return_iterator=return_iterator,
        filter_fn=self.filter_fn if self.subset == 'train' else None,
        balance_pos_neg=balance_pos_neg,
        pos_filter_fn=self.pos_filter_fn if self.subset == 'train' else None,
        neg_filter_fn=self.neg_filter_fn if self.subset == 'train' else None,
        count_fn=self.count_fn if self.subset == 'train' else None,
        name=self.subset) 

  @staticmethod
  def num_examples_per_epoch(subset='train', dir=None):
    default_value = 10000
    if subset == 'train':
      file = (dir or os.path.dirname(FLAGS.train_input.split(',')[0])) + '/num_records.txt'
      return gezi.read_int_from(file, default_value)
    elif subset == 'valid':
      file = (dir or os.path.dirname(FLAGS.valid_input)) + '/num_records.txt'
      return gezi.read_int_from(file, default_value)
    elif subset == 'test':
      file = (dir or os.path.dirname(FLAGS.test_input)) + '/num_records.txt'
      return gezi.read_int_from(file, default_value)
    else:
      raise ValueError('Invalid data subset "%s"' % subset)

