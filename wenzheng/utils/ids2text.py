#!/usr/bin/env python
# -*- coding: gbk -*-
# ==============================================================================
#          \file   text2ids.py
#        \author   chenghuige  
#          \date   2016-08-29 15:26:15.418566
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import sys
import gezi

from wenzheng.utils import vocabulary 
#TODO-- remove conf.py using gfalgs or yaxml

try:
  import conf
  from conf import TEXT_MAX_WORDS, ENCODE_UNK
except Exception:
  print('Warning: no conf.py in current path use util conf', file=sys.stderr)
  from wenzheng.utils.conf import TEXT_MAX_WORDS, ENCODE_UNK

vocab = None 

NUM_MARK = '<NUM>'

def init(vocab_path=None):
  global vocab, Segmentor
  if vocab is None:
    vocabulary.init(vocab_path)
    print('ENCODE_UNK', ENCODE_UNK, file=sys.stderr)
    vocab = vocabulary.get_vocab()

def ids2words(text_ids, print_end=True):
  #print('@@@@@@@@@@text_ids', text_ids)
  #NOTICE int64 will be ok
#  Boost.Python.ArgumentError: Python argument types in
#    Identifer.key(Vocabulary, numpy.int32)
#did not match C++ signature:
#    key(gezi::Identifer {lvalue}, int id, std::string defualtKey)
#    key(gezi::Identifer {lvalue}, int id)
  #words = [vocab.key(int(id)) for id in text_ids if id > 0 and id < vocab.size()]
  words = []
  for id in text_ids:
    if id > 0 and id < vocab.size():
      #@NOTICE! must has end id, @TODO deal with UNK word
      if id != vocab.end_id():
        word = vocab.key(int(id))
        words.append(word)
      else:
        if print_end:
          words.append('</S>')
        break
    else:
      break
  return words

def ids2text(text_ids, sep=' ', print_end=True):
  return sep.join(ids2words(text_ids, print_end=print_end))

def idslist2texts(text_ids_list, sep=' ', print_end=True):
  return [ids2text(text_ids, sep=sep, print_end=print_end) for text_ids in text_ids_list]
  #return [sep.join([vocab.key(int(id)) for id in text_ids if id > 0 and id < vocab.size()]) for text_ids in text_ids_list]

def translate(text_ids):
  return ids2text(text_ids, sep='', print_end=False)

def translates(text_ids_list):
  return [translate(text_ids) for text_ids in text_ids_list]

def start_id():
  return vocab.end_id()

def end_id():
  return vocab.end_id()

def unk_id():
  return vocab.unk_id()
