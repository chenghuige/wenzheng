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

  def make_batch(self, batch_size, filenames=None):
    """Read the images and labels from 'filenames'."""
    #with tf.device('/cpu:0'):
    filenames = filenames or self.get_filenames()
    logging.info(self.subset, 'num files', len(filenames))
    assert filenames, self.subset
    min_queue_examples = 20000
    repeat = False if tf.executing_eagerly() else True
    if self.subset == 'train':
      return melt.dataset_decode.inputs(
        filenames, 
        decode_fn=self.parser,
        batch_size=batch_size,
        num_threads=FLAGS.num_threads,
        shuffle_files=True,
        fix_sequence=False,
        buffer_size=min_queue_examples + 3 * batch_size,
        initializable=True,
        repeat=repeat,
        name=self.subset)
    else:
      return melt.dataset_decode.inputs(
        filenames, 
        decode_fn=self.parser,
        batch_size=batch_size,
        shuffle_files=False,
        num_threads=FLAGS.num_threads,
        fix_sequence=True,
        buffer_size=min_queue_examples + 3 * batch_size,
        initializable=True,
        repeat=repeat,
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

