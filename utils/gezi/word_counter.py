#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   word_counter.py
#        \author   chenghuige  
#          \date   2018-01-17 11:06:31.146818
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
from collections import Counter

class WordCounter(object): 
  def __init__(self,
               most_common=None, 
               min_count=None,
               write_unknown=True,
               unknown_mark='<UNK>'):
    self.most_common = most_common
    self.min_count = min_count or 0
    self.write_unknown = write_unknown
    self.unknown_mark = unknown_mark

    self.counter = Counter()
    self.total = 0

  def add(self, word, count=1):
    self.counter[word] += count 
    self.total += count 
  
  def save(self, filename, most_common=None, min_count=None):
    if not most_common:
      most_common = self.most_common
      if not most_common:
        most_common = len(self.counter)
    if not min_count:
      min_count = self.min_count

    with open(filename, 'w') as out:
      l = []
      total = 0
      for word, count in self.counter.most_common(most_common):
        if count < min_count:
          break
        l.append((word, count))
        total += count 

      if self.write_unknown:
        unknown_count = self.total - total 
      else:
        unknown_count = 0

      print('total_count', self.total, 'unknown_count', unknown_count, 'total_word', len(l), file=sys.stderr)

      for word, count in l:
        if unknown_count > count:
          print(self.unknown_mark, unknown_count, sep='\t',file=out)
          unknown_count = 0
        print(word, count, sep='\t', file=out)


      
