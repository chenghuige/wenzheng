#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   show-records.py
#        \author   chenghuige  
#          \date   2018-03-29 21:55:32.616016
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import tensorflow as tf

input = sys.argv[1]

for example in tf.python_io.tf_record_iterator(input):
  result = tf.train.Example.FromString(example)
  print(result)
  break
