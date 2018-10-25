#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-mix-vocab.py
#        \author   chenghuige  
#          \date   2018-10-23 06:48:14.182012
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

vocab_file = sys.argv[1] 
char_vocab_file = sys.argv[2]
num_words = int(sys.argv[3]) 

words = set()
for i, line in enumerate(open(vocab_file)):
  if i == num_words:
    break
  word, count = line.rstrip('\n').split('\t')

  words.add(word)
  print(word, count, sep='\t')

for line in open(char_vocab_file):
  ch, count = line.rstrip('\n').split('\t')
  if ch not in words:
    print(ch, count, sep='\t')

