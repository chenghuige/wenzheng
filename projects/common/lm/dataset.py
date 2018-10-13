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

flags.DEFINE_integer('bptt', 70, '')

import gezi
import melt
logging = melt.logging

import numpy as np

import gezi

class Dataset(melt.tfrecords.Dataset):
  def __init__(self, subset='train'):
    super(Dataset, self).__init__(subset)
    self.epoch_size = None
    self.batch_size = None
    self.data_npy = None
    self.data = None

  # https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/reader.py
  # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/rnn_ptb/rnn_ptb.py
  # TODO no shuffle each epoch here
  def make_batch(self, batch_size, filenames, bptt=None, **kwargs):
    bptt = bptt or FLAGS.bptt
    self.batch_size = batch_size
    if self.data is None:
      data_npy = np.load(filenames[0])
      self.data = data_npy
      self.data_npy = data_npy
      self.epoch_size = ((len(np.concatenate(self.data_npy)) // batch_size) - 1) // bptt
    else:
      data_npy = self.data_npy

    with tf.device('/cpu:0'):
      if not tf.executing_eagerly():
        if self.data is None:
          self.data_npy_ori = data_npy
          self.data_npy = np.concatenate(data_npy)
          data_npy = self.data_npy

        data_npy = self.data_npy
        if self.data is None:
          #self.data = tf.get_variable('input_%s' % self.subset, dtype=tf.int32, shape=data_npy.shape, initializer=tf.constant_initializer(data_npy), trainable=False)
          self.data = tf.get_variable('input_%s' % self.subset, dtype=tf.int32, shape=data_npy.shape,trainable=False)
          data_placeholder = tf.placeholder(tf.int32, data_npy.shape)
          data_init = self.data.assign(data_placeholder)
          sess = melt.get_session()
          sess.run(data_init, feed_dict={data_placeholder: data_npy})

        data = self.data

        data_len = tf.size(data)
        batch_len = data_len // batch_size
        data = tf.reshape(data[:batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // bptt
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or bptt")
        with tf.control_dependencies([assertion]):
          epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, 
                            [0, i * bptt],
                            [batch_size, (i + 1) * bptt])
        x.set_shape([batch_size, bptt])
        y = tf.strided_slice(data, 
                            [0, i * bptt + 1],
                            [batch_size, (i + 1) * bptt + 1])
        y.set_shape([batch_size, bptt])

        class Iter(object):
          def __init__(self, x, y):
            self.x = x 
            self.y = y
          
          def __iter__(self):
            return self

          def get_next(self):
            return self.x, self.y
      
        iter = Iter(x, y)
        return iter
      else:
        # in eager mode if tf.get_variable will be very slow...
        # epoch:0.02/1024 step:8600 elapsed:[1.312] batch_size:[32] batches/s:[76.23] insts/s:[2439] 1epoch:[1.40h] lr:[0.0010000] train_loss:[5.0710] valid_loss:[5.0503]
        class Iter():
          def __init__(self, data):
            self.ori_data = data
            self.reset()

          def reset(self):
            self.i = 0
            np.random.shuffle(self.ori_data)
            self.data = np.concatenate(self.ori_data)
            data_len = len(self.data)
            batch_len = data_len // batch_size
            self.data = self.data[:batch_size * batch_len].reshape([batch_size, batch_len])

          def __iter__(self):
            return self

          def __next__(self):
            i = self.i
            data = self.data
          
            if i < data.shape[1]:
              slen = min(bptt, data.shape[1] - 1 - i)
              x = data[:, i:i + slen]
              y = data[:, i + 1:i + 1 + slen]
              self.i += bptt
              return x, y
            else:
              self.reset()
              raise StopIteration()

        return Iter(data_npy)



  def num_examples_per_epoch(self, mode):
    return self.epoch_size * self.batch_size
