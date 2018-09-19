#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   hash.py
#        \author   chenghuige  
#          \date   2018-04-28 12:04:55.557328
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import hashlib
import six 
import numpy as np

def hash_str(input):
  if not six.PY2:
    return hex(int(hashlib.sha256(input.encode('utf8')).hexdigest(), 16) % sys.maxsize)[2:]
  else:
  	# ignore last L
  	return hex(int(hashlib.sha256(input.encode('utf8')).hexdigest(), 16) % sys.maxsize)[2:-1]

# NOTICE ! fasttext use uint32_t !
def fasttext_hash(word):
  h = 2166136261
  for w in word:
    h = np.uint32(h ^ ord(w))
    h = np.uint32(h * 16777619)
  return h

hash = fasttext_hash

