#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   ngram.py
#        \author   chenghuige  
#          \date   2018-03-01 11:25:51.105094
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import numpy as np

# def get_ngrams(input, minn=3, maxn=3, start='<', end='>'):
#   input = start + input + end   
#   len_ = len(input)
#   ngrams = []
#   for i in range(0, len_ - minn + 1):
#     for j in range(i + minn, i + maxn + 1):
#       if j <= len_:
#         ngrams.append(input[i:j])

#   return ngrams

def get_ngrams(input, minn=3, maxn=3, start='<', end='>'):
  input = start + input + end   
  len_ = len(input)
  ngrams = []
  for ngram in reversed(range(minn, maxn + 1)):
    for i in range(0, len_ - ngram + 1):
      ngrams.append(input[i:i + ngram])

  return ngrams

from gezi import hash  

# defaut 3, 6 according to fasttext default ngram, but may use 3, 3 only trigram 
def get_ngrams_hash(input, buckets, minn=3, maxn=6, start='<', end='>', reserve=0):
  ngrams = get_ngrams(input, minn, maxn, start, end)
  #print(ngrams)
  ngrams = [reserve + hash(x) % buckets for x in ngrams]
  return ngrams

def fasttext_ids(word, vocab, buckets, minn=3, maxn=6, start='<', end='>'):
  ngrams = get_ngrams(word, minn, maxn, start, end)
  ngram_ids = [vocab.size() + hash(x) % buckets for x in ngrams]
  if vocab.has(word):
    ids = [vocab.id(word)] + ngram_ids
  else:
    ids = ngram_ids
  return ids
  

