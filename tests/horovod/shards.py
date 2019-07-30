#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   shards.py
#        \author   chenghuige  
#          \date   2019-07-30 17:55:17.754386
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf

dataset = tf.data.Dataset.range(6)
dataset = dataset.shard(FLAGS.num_workers, FLAGS.worker_index)


iterator = dataset.make_one_shot_iterator()
res = iterator.get_next()

# Suppose you have 3 workers in total
with tf.Session() as sess:
    for i in range(2):
        print(sess.run(res)) 
