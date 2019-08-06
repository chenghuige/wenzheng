#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   torch-hvd-test.py
#        \author   chenghuige  
#          \date   2019-08-06 11:39:03.209243
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

from torch.utils.data import DataLoader
import gezi
import lele

import melt
melt.init()
#import horovod.tensorflow as hvd
#import horovod.tensorflow as hvd

# from pyt.dataset import *
# from text_dataset import Dataset as TD

import numpy as np

#import horovod.tensorflow as hvd
import horovod.torch as hvd
hvd.init()

for i in range(5):
  print(i)  
