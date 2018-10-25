#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-char-vocab.py
#        \author   chenghuige  
#          \date   2018-09-11 20:25:15.703398
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

for line in sys.stdin:
  word, freq = line.rstrip('\n').split('\t')
  if (word.startswith('<') and word.endswith('>')) or len(word) == 1:
    print(line, end='')
