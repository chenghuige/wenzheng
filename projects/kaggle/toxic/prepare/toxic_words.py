#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   badwords.py
#        \author   chenghuige  
#          \date   2018-03-08 15:57:34.430100
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import pandas as pd

# import melt 
# logging = melt.logging

import six 
#assert six.PY3

# https://www.freewebheaders.com/full-list-of-bad-words-banned-by-google/
google_banned_words_file = '/home/gezi/data/kaggle/toxic/full-list-of-bad-words-banned-by-google-txt-file_2013_11_26_04_53_31_867.txt'  

# https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/lexicons
hate_words_file = '/home/gezi/data/kaggle/toxic/hate-speech-and-offensive-language-master/lexicons/hatebase_dict.csv' 

toxic_words = None

def get_toxic_words():
  global toxic_words
  if toxic_words is None:
    toxic_words = set()
    # google banned
    for line in open(google_banned_words_file, encoding='utf-8', errors='ignore'):
      toxic_words.add(line.strip())

    #logging.info('google len(toxic_words):', len(toxic_words))

    df = pd.read_csv(hate_words_file)
    l = df.values
    l = [x[0].strip(',').strip("'") for x in l]
    for word in l:
      toxic_words.add(word)
    #logging.info('add hate:', len(toxic_words))

  return toxic_words


if __name__ == '__main__':
  l#ogging.set_logging_path('~/tmp')
  toxic_words = get_toxic_words()
  print(toxic_words)       
