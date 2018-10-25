#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   check.py
#        \author   chenghuige  
#          \date   2018-10-20 23:58:33.339526
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

import gezi
import numpy as np

m = {}
for line in open('./vectors.fix.txt'):
  l = line.strip().split() 
  if len(l) < 100:
    continue
  key = l[0]
  vec = np.array([float(x) for x in l[1:]])

  m[key] = vec 

def sim(x, y):
  print(x, y, gezi.cosine(m[x], m[y]))


sim('女人', '女孩')
sim('女人', '今天')
sim('征途', '征程')
sim('征途', '电脑')
sim('笔记本', '电脑')

def main(_):
  pass

if __name__ == '__main__':
  tf.app.run()  
  
