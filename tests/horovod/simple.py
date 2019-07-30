#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   simple.py
#        \author   chenghuige  
#          \date   2019-07-30 16:26:23.861318
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf
import horovod.tensorflow as hvd

tf.enable_eager_execution()
hvd.init()

dataset = tf.data.Dataset.range(200).repeat()
dataset = dataset.shard(hvd.size(), hvd.rank())
dataset = dataset.batch(64, drop_remainder=True)

i = 0
for d in dataset:
  i += 1

print("batches on rank %d: %d" % (hvd.rank(), i))  
