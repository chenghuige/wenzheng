#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   fix.py
#        \author   chenghuige  
#          \date   2018-10-20 23:43:54.852568
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

words = []
for line in open('./vocab.txt'):
  try:
    words.append(line.rstrip('\n').split()[0])
  except Exception:
    print(line, file=sys.stderr)

for i, line in enumerate(open('./vectors.txt')): 
  _, vec = line.rstrip().split(' ', 1)
  if i < len(words):
    print(words[i], vec, sep=' ')
  else:
    print(line)


