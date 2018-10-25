#!/usr/bin/env python
# -*- coding: utf8 -*-
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
##well ids2text use it's own vocab..
#from wenzheng.utils.ids2text import ids2words, ids2text, idslist2texts, translate, translates

try:
  import conf
  from conf import TEXT_MAX_WORDS, ENCODE_UNK
  try:
    from conf import MULTI_GRID 
  except Exception:
    MULTI_GRID = True
except Exception:
  print('Warning: no conf.py in current path use util conf', file=sys.stderr)
  from wenzheng.utils.conf import TEXT_MAX_WORDS, ENCODE_UNK
  try:
    from wenzheng.utils.conf import MULTI_GRID 
  except Exception:
    MULTI_GRID = True

vocab = None 
Segmentor = None 

NUM_MARK = '<NUM>'
EN_MRAK = '<EN>'

def init(vocab_path=None, append=None):
  global vocab, Segmentor
  if vocab is None:
    vocabulary.init(vocab_path, append=append)
    print('ENCODE_UNK', ENCODE_UNK, file=sys.stderr)
    vocab = vocabulary.get_vocab()
    Segmentor = gezi.Segmentor()

def get_id(word, unk_vocab_size=None):
  if unk_vocab_size and not vocab.has(word):
    return gezi.hash(word) % unk_vocab_size + vocab.size()
  return vocab.id(word)
  
#@TODO gen-records should use text2ids
#TODO ENCODE_UNK might not be in conf.py but to pass as param encode_unk=False
def words2ids(words, feed_single=True, allow_all_zero=False, 
              pad=True, append_start=False, append_end=False,
              max_words=None, norm_digit=True, norm_all_digit=False,
              multi_grid=None, encode_unk=None, feed_single_en=False,
              digit_to_chars=False,
              unk_vocab_size=None):
  """
  default params is suitable for bow
  for sequence method may need seg_method prhase and feed_single=True,
  @TODO feed_single is for simplicity, the best strategy may be try to use one level lower words
  like new-word -> phrase -> basic -> single cn

  #@TODO feed_single move to Segmentor.py to add support for seg with vocab 
  norm_all_digit is not used mostly, since you can control this behavior when gen vocab 
  """
  multi_grid = multi_grid or MULTI_GRID
  encode_unk = encode_unk or ENCODE_UNK
    
  new_words = []
  if not feed_single:
    word_ids = [get_id(word, unk_vocab_size) for word in words if vocab.has(word) or encode_unk]
  else:
    word_ids = []
    for word in words:
      if digit_to_chars and any(char.isdigit() for char in word):
        for w in word:
          if not vocab.has(w) and unk_vocab_size:
              word_ids.append(gezi.hash(w) % unk_vocab_size + vocab.size())
              new_words.append(w)
          else:
            if vocab.has(w) or encode_unk:
              word_ids.append(vocab.id(w))
              new_words.append(w)
        continue
      elif norm_all_digit and word.isdigit():
        word_ids.append(vocab.id(NUM_MARK))
        new_words.append(word)
        continue
      if vocab.has(word):
        word_ids.append(vocab.id(word))
        new_words.append(word)
      elif not norm_all_digit and norm_digit and word.isdigit():
        word_ids.append(vocab.id(NUM_MARK))
        new_words.append(word)
      else:
        #TODO might use trie to speed up longest match segment
        if (not multi_grid) or feed_single_en:
          if not feed_single_en:
            chars = gezi.get_single_cns(word)
          else:
            chars = word
          if chars:
            for w in chars:
              if not vocab.has(w) and unk_vocab_size:
                word_ids.append(gezi.hash(w) % unk_vocab_size + vocab.size())
                new_words.append(w)
              else:
                if vocab.has(w) or encode_unk:
                  word_ids.append(vocab.id(w))
                  new_words.append(w)
          else:
            if unk_vocab_size:
              word_ids.append(gezi.hash(word) % unk_vocab_size + vocab.size())
              new_words.append(word)
            else:
              if encode_unk:
                word_ids.append(vocab.unk_id())
                new_words.append(word)
        else:
          #test it!  print text2ids.ids2text(text2ids.text2ids('匍匐前进'))
          word_ids += gezi.loggest_match_seg(word, vocab, encode_unk=encode_unk, unk_vocab_size=unk_vocab_size, vocab_size=vocab.size())
          # NOTICE new_words lost here!

  if append_start:
    word_ids = [vocab.start_id()] + word_ids

  if append_end:
    word_ids = word_ids + [vocab.end_id()]

  if not allow_all_zero and  not word_ids:
    word_ids.append(vocab.end_id())

  if pad:
    word_ids = gezi.pad(word_ids, max_words or TEXT_MAX_WORDS, 0)  

  return word_ids, new_words

# TODO might change to set pad default as False
def text2ids(text, seg_method='basic', feed_single=True, allow_all_zero=False, 
            pad=True, append_start=False, append_end=False, to_lower=True,
            max_words=None, norm_digit=True, norm_all_digit=False,
            multi_grid=None, remove_space=True, encode_unk=None, feed_single_en=False,
            digit_to_chars=False, unk_vocab_size=None, return_words=False):
  """
  default params is suitable for bow
  for sequence method may need seg_method prhase and feed_single=True,
  @TODO feed_single is for simplicity, the best strategy may be try to use one level lower words
  like new-word -> phrase -> basic -> single cn

  #@TODO feed_single move to Segmentor.py to add support for seg with vocab 
  """
  if to_lower:
    text = text.lower()
  words = Segmentor.Segment(text, seg_method)
  if remove_space:
    #words = [x for x in words if x.strip()]
    # change to remove duplicate space
    words = [x for i, x in enumerate(words) if not (i < len(words) - 1 and not(x.strip()) and not(words[i + 1].strip()))]

  ids, new_words = words2ids(words, 
                            feed_single=feed_single, 
                            allow_all_zero=allow_all_zero, 
                            pad=pad, 
                            append_start=append_start, 
                            append_end=append_end, 
                            max_words=max_words, 
                            norm_digit=norm_digit,
                            norm_all_digit=norm_all_digit,
                            multi_grid=multi_grid,
                            encode_unk=encode_unk,
                            feed_single_en=feed_single_en,
                            digit_to_chars=digit_to_chars,
                            unk_vocab_size=unk_vocab_size)
  if not return_words:
    return ids 
  else:
    return ids, new_words

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

def text2segtext(text, seg_method='basic', feed_single=True, allow_all_zero=False, pad=True, sep=' '):
  return ids2text(text2ids(text, seg_method=seg_method, feed_single=feed_single, allow_all_zero=allow_all_zero))

def texts2segtexts(texts, seg_method='basic', feed_single=True, allow_all_zero=False, pad=True, sep=' '):
  return idslist2texts(texts2ids(texts,seg_method=seg_method, feed_single=feed_single, allow_all_zero=allow_all_zero))

def segment(text, seg_method='basic'):
  return Segmentor.Segment(text, seg_method=seg_method)

def texts2ids(texts, seg_method='basic', feed_single=True, allow_all_zero=False, pad=True):
  return np.array([text2ids(text, seg_method, feed_single, allow_all_zero, pad) for text in texts])

def start_id():
  return vocab.end_id()

def end_id():
  return vocab.end_id()

def unk_id():
  return vocab.unk_id()


#TODO duplicate with ids2text
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
      #break
      words.append('<UNK:{}>'.format(id))
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
