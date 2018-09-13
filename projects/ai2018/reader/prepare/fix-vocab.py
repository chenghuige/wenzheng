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
for line in sys.stdin:
  word, count = line.rstrip('\n').split('\t')
  words.add(word)
  print(word, count, sep='\t')

chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
chars2 = [x.upper() for x in chars]
chars += chars2 

chars = [x for x in chars if x not in words]

for word in chars:
  print(word, 1, sep='\t')

