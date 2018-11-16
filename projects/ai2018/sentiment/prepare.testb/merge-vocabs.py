#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   merge-vocabs.py
#        \author   chenghuige  
#          \date   2018-10-23 02:07:23.291646
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

m = {}

files = sys.argv[1:]
num_files = len(files)

print('num_files', num_files, file=sys.stderr)

for file_ in files:
  for line in open(file_):
    word, count = line.rstrip('\n').split('\t', 1)
    count = int(count)
    if word not in m:
      m[word] = count 
    else:
      m[word] += count

sorted_by_value = sorted(m.items(), key=lambda kv: -kv[1])

for key, val in sorted_by_value:
  print(key, val, sep='\t')

