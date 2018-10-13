#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   read-records.py
#        \author   chenghuige  
#          \date   2018-10-11 16:44:51.910748
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

#flags.DEFINE_string('input', '', '')
#flags.DEFINE_string('output', '', '')

import sys 
import os

import melt

import dataset

FLAGS.valid_input = './mount/temp/lm/corpus/sentiment/valid/ids.npy'

def main(_):
  melt.apps.init()
  if not tf.executing_eagerly():
    ds = dataset.Dataset('valid')
    iter = ds.make_batch(32, [FLAGS.valid_input], bptt=70)
    x, y = iter.get_next()

    def run(sess, step):
      x_, y_ = sess.run([x, y])
      #if step == 210:
      print(step, x_, y_)
      # print(len(x_), len(x_[0]))
      # exit(0)
    melt.flow.tf_flow(run)
  else:
    ds = dataset.Dataset('valid')
    iter = ds.make_batch(32, [FLAGS.valid_input], bptt=70)
    for i, (x, y) in enumerate(iter):
      #if i == 210:
      print(i, x, y)
      #exit(0)


if __name__ == '__main__':
  tf.app.run()  
  
