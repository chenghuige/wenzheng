#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   write.py
#        \author   chenghuige  
#          \date   2016-08-24 10:21:46.629992
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import random
import tensorflow as tf
 
class Writer(object):
  def __init__(self, file, buffer_size=None):
    random.seed(12345)
    self.count = 0
    self.buffer_size = buffer_size
    self.writer = tf.python_io.TFRecordWriter(file)
    self.buffer = [] if self.buffer_size else None

  def __del__(self):
    #print('del writer', file=sys.stderr)
    self.close()

  def __enter__(self):
    return self  

  def __exit__(self, exc_type, exc_value, traceback):
    #print('close writer', file=sys.stderr)
    self.close()

  def close(self):
    if self.buffer:
      random.shuffle(self.buffer)
      for example in self.buffer:
        self.writer.write(example.SerializeToString())
      self.buffer = []   

  def finalize(self):
    self.close()

  def write(self, example):
    self.count += 1
    if self.buffer is not None:
      self.buffer.append(example)
      if len(self.buffer) >= self.buffer_size:
        random.shuffle(self.buffer)
        for example in self.buffer:
          self.writer.write(example.SerializeToString())
        self.buffer = []
    else:
      self.writer.write(example.SerializeToString())

  def size(self):
    return self.count


class MultiOutWriter(object):
  """
  read single file output to mutlitple tfrecord
  """
  def __init__(self, dir, name='train', max_lines=50000):
     self.dir = dir
     self.name = name 
     self.max_lines = max_lines
     self.index = 0
     self.count = 0
     self.writer = self.get_tfrecord()
  
  def __del__(self):
    print('del writer', file=sys.stderr)
    self.close()

  def __enter__(self):
    return self  

  def __exit__(self, exc_type, exc_value, traceback):
    print('close writer', file=sys.stderr)
    self.close()

  def get_tfrecord(self):
    return tf.python_io.TFRecordWriter(
      os.path.join(dir, '{}_{}'.format(self.name, self.index)))
  
  def write(self, example):
    self.writer.write(example.SerializeToString())
    self.count += 1
    if self.count == self.max_lines:
      self.index += 1
      self.writer.close()
      self.writer = self.get_tfrecord()


import numpy as np
class RandomSplitWriter(object):
  """
  read single file, random split as train, test to two files
  """
  def __init__(self, train_file, test_file, train_ratio=0.8):
    self.train_writer = Writer(train_file)
    self.test_writer = Writer(test_file)
    self.train_ratio = train_ratio

  def __enter__(self):
    return self  

  def __del__(self):
    print('del writer', file=sys.stderr)
    self.close()

  def __exit__(self, exc_type, exc_value, traceback):
    print('close writer', file=sys.stderr)
    self.close()
    
  def close(self):
    self.train_writer.close()
    self.test_writer.close()

  def write(example):
    writer = self.train_writer if np.random.random_sample() < self.train_ratio else self.test_writer()
    writer.write(example)

class RandomSplitMultiOutWriter(object):
  """
  read single file, random split as train, test each to mulitple files
  """
  def __init__(self, train_dir, test_dir, train_name='train', test_name='test', max_lines=50000, train_ratio=0.8):
    self.train_writer = MultiOutWriter(train_dir, train_name, max_lines)
    self.test_writer = MultiOutWriter(test_dir, test_name, max_lines)
    self.train_ratio = train_ratio

  def __enter__(self):
    return self  

  def __del__(self):
    print('del writer', file=sys.stderr)
    self.close()

  def __exit__(self, exc_type, exc_value, traceback):
    print('close writer', file=sys.stderr)
    self.close()

  def close(self):
    self.train_writer.close()
    self.test_writer.close()

  def write(self, example):
    writer = self.train_writer if np.random.random_sample() < self.train_ratio else self.test_writer()
    writer.write(example)

