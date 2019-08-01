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

# NOTICE if batc_parse, first batch then map.. you have faster speed but each batch is the same size like 256 even for the last batch with drop remind=False
# have tested textlinedataset behavior like above TODO check tfrecord
class Dataset(object):
  def __init__(self, 
               subset='train',
               batch_size=None,
               filenames=None, 
               InputDataset=None, 
               batch_parse=False,
               hvd_shard=True):
    self.subset = subset
    self.filter_fn = None
    self.pos_filter_fn = None
    self.neg_filter_fn = None 
    self.count_fn = None
    self.InputDataset = InputDataset
    self.batch_parse = batch_parse
    self.batch_size = batch_size or FLAGS.batch_size
    self.filenames = filenames or self.get_filenames()
    self.hvd_shard = hvd_shard

  def get_filenames(self):
    try:
      if self.subset in ['train', 'valid', 'test']:
        if self.subset == 'train':
          return gezi.list_files(FLAGS.train_input)
        elif self.subset == 'valid':
          return gezi.list_files(FLAGS.valid_input)
        elif self.subset == 'test':
          return gezi.list_files(FLAGS.test_input)
      else:
        raise ValueError('Invalid data subset "%s"' % self.subset)
    except Exception:
      return None

  def parser(self, example):
    pass

  def adjust(self, result):
    return result

  def make_batch(self, 
                 batch_size=None, 
                 filenames=None,
                 initializable=False,
                 repeat=None,
                 return_iterator=True,
                 hvd_shard=None,
                 simple_parse=False):
    """Read the images and labels from 'filenames'."""
    #with tf.device('/cpu:0'):
    hvd_shard = hvd_shard if hvd_shard is not None else self.hvd_shard
    batch_size = batch_size if batch_size is not None else self.batch_size
    self.batch_size = batch_size
    filenames = filenames if filenames is not None else self.filenames
    logging.info(self.subset, 'num files', len(filenames))
    assert filenames, self.subset
    min_queue_examples = 20000
    allow_smaller_final_batch = True
    if repeat is None:
      if tf.executing_eagerly():
        repeat = False # if True will not consider epoch stop using for... loop forever for item in dataset..
      else:
        # for eval in num_gpus > 1 then set repeat = True so final batch with full batch
        # TODO 
        num_gpus = melt.num_gpus() if not 'OMPI_COMM_WORLD_RANK' in os.environ else 1
        if self.subset == 'train' or num_gpus > 1:
          repeat = True
        else:
          repeat = False

    if self.subset == 'train':
      shuffle_files=True 
      fix_sequence = False
      # if self.batch_parse:
      #   allow_smaller_final_batch = False
    else:
      shuffle_files = False
      fix_sequence = True
      # TODO try horovod metric evaluate using multiple gpu

    balance_pos_neg=False
    if self.pos_filter_fn and self.neg_filter_fn:
      balance_pos_neg = True

    print('-----------dataset repeat', repeat)
    print('-----------dataset batch_parse', self.batch_parse)
    print('-----------dataset allow final small batch', allow_smaller_final_batch)
    # for bow using cpu 69 insts/s using gpu 54 inst/s
    with tf.device('/cpu:0'):
      result = melt.dataset_decode.inputs(
        filenames, 
        decode_fn=self.parse,
        batch_size=batch_size,
        num_threads=FLAGS.num_threads,
        shuffle_files=shuffle_files,
        fix_sequence=fix_sequence,
        buffer_size=min_queue_examples + 3 * batch_size if not FLAGS.buffer_size else FLAGS.buffer_size,
        initializable=initializable,
        repeat=repeat,
        allow_smaller_final_batch=allow_smaller_final_batch,
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
        name=self.subset,
        Dataset=self.InputDataset,
        batch_parse=self.batch_parse,
        hvd_shard=hvd_shard,
        training=self.subset == 'train',
        simple_parse=simple_parse) 

      return self.adjust(result)


  @staticmethod
  def num_examples_per_epoch(subset='train', dir=None):
    default_value = None
    if subset == 'train':
      file = (dir or gezi.dirname(FLAGS.train_input.split(',')[0])) + '/num_records.txt'
      return gezi.read_int_from(file, default_value)
    elif subset == 'valid':
      file = (dir or gezi.dirname(FLAGS.valid_input)) + '/num_records.txt'
      return gezi.read_int_from(file, default_value)
    elif subset == 'test':
      file = (dir or gezi.dirname(FLAGS.test_input)) + '/num_records.txt'
      return gezi.read_int_from(file, default_value)
    else:
      raise ValueError('Invalid data subset "%s"' % subset)

