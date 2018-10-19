#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   fix-vocab.py
#        \author   chenghuige  
#          \date   2018-09-13 14:49:43.197787
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

words = set()
for line in open(sys.argv[1]):
  word, count = line.rstrip('\n').split('\t')
  words.add(word)
  print(word, count, sep='\t')

for line in open(sys.argv[2]):
  word, count = line.rstrip('\n').split('\t')
  if word not in words:
    print(word, 1, sep='\t')

