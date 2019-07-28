#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   read.py
#        \author   chenghuige  
#          \date   2019-07-26 18:02:22.038876
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from tfrecord_dataset import Dataset

import tensorflow as tf 
tf.enable_eager_execution()


dataset = Dataset('train')

da = dataset.make_batch()

for i, batch in enumerate(da):
    print('---------------------------', i)
    print(batch)
    #print(batch[0]['index'][0])
    #print(batch[0]['field'][0])
    if i == 2:
        exit(0)
exit(0)
