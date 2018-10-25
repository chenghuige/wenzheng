#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   seg2corpus.py
#        \author   chenghuige  
#          \date   2018-10-20 08:10:01.731952
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

for line in sys.stdin:
  l = line.rstrip('\n').split('\t', 1)[1].split('\x09')
  if not l:
    continue
  if '|' in l[0]:
    l = [x.split('|')[0] for x in l]
  print(' '.join(l))

def main(_):
  pass

if __name__ == '__main__':
  tf.app.run()  
  
