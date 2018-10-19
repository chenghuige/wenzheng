#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   tfrecords.py
#        \author   chenghuige  
#          \date   2018-10-17 13:36:30.235928
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import random  

class Writer(object):
  def __init__(self, file, buffer_size=None):
    import tensorflow as tf
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
