#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   tokenize-corpus.py
#        \author   chenghuige  
#          \date   2018-03-02 22:42:15.222451
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import multiprocessing as mp

import gezi

import glob

def tokenize(file_):
  with open(file_.replace('chunk', 'tokens'), 'w') as out:
    first = True
    for line in open(file_):
      if first:
        first = False
        continue
      l = line.rstrip().split('\t')
      comment = l[1].replace('REDIRECT Talk:', '').replace('NEWLINE:*', '').replace('NEWLINE*', ' ').replace('NEWLINE', ' ').replace('TAB', ' ').replace('"', ' ')
      comment = comment.lstrip(': ')
      tokens = gezi.segment.tokenize_filter_empty(comment)
      print(' '.join(tokens), file=out)


#files = glob.glob('/home/gezi/data/kaggle/toxic/talk_corpus/comments*/chunk*')
files = glob.glob('/home/gezi/data/kaggle/toxic/talk_corpus/comments_user_2015/chunk*')

pool = mp.Pool()
pool.map(tokenize, files)
pool.close()
pool.join()
