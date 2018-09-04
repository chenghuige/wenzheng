#!/usr/bin/env python
# ==============================================================================
#          \file   read.py
#        \author   chenghuige  
#          \date   2016-08-17 10:30:16.621213
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import melt

flags = tf.app.flags
#import gflags as flags
FLAGS = flags.FLAGS

'''
deprecated not used
'''

flags.DEFINE_integer('num_epochs', 0, 'Number of epochs to run trainer.')
flags.DEFINE_integer('num_preprocess_threads', 12, '')
flags.DEFINE_boolean('batch_join', True, '')
flags.DEFINE_boolean('shuffle', True, '')
flags.DEFINE_integer('min_after_dequeue', 20000, '')

def sparse_inputs(files, decode, batch_size=64):
  return melt.tfrecords.read_sparse.inputs(files, 
                                           decode, 
                                           batch_size=batch_size, 
                                           num_epochs=FLAGS.num_epochs,
                                           num_preprocess_threads=FLAGS.num_preprocess_threads, 
                                           shuffle=FLAGS.shuffle,
                                           batch_join=FLAGS.batch_join,
                                           min_after_dequeue=FLAGS.min_after_dequeue)

def inputs(files, decode, batch_size=64):
  return melt.tfrecords.read.inputs(files, 
                                    decode, 
                                    batch_size=batch_size, 
                                    num_epochs=FLAGS.num_epochs,
                                    num_preprocess_threads=FLAGS.num_preprocess_threads, 
                                    shuffle=FLAGS.shuffle,
                                    batch_join=FLAGS.batch_join,
                                    min_after_dequeue=FLAGS.min_after_dequeue)

