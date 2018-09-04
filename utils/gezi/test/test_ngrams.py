#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   test_ngrams.py
#        \author   chenghuige  
#          \date   2018-03-01 11:43:55.445657
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import gezi

import six

assert six.PY3

def ngrams(word, minn=3, maxn=3):
  print(word)
  print(gezi.get_ngrams(word))
  print(gezi.get_ngrams(word, 2, 3))
  print(gezi.get_ngrams(word, 2, 6), len(gezi.get_ngrams(word, 2, 6)))
  print(gezi.get_ngrams_hash(word, 1000))
  
ngrams('motherfuck')
ngrams('我的天啊')
ngrams('FrenchMan')
