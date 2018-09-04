#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   normalize.py
#        \author   chenghuige  
#          \date   2018-02-13 19:51:49.324339
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

import sys, os

from collections import namedtuple
import gezi

import re

try:
  import toxic_words
except Exception:
  import prepare.toxic_words

# TODO...
try:
  from preprocess import *
except Exception:
  from prepare.preprocess import *

ip_pattern = r"(\d+\.\d+\.\d+\.\d+)"
http_pattern = r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]"

# !NOTICE can not set file name to tokenize, will confict with python3 some tokenize.py 
vocab = None

# this vocab is original calculated word freq vocab just using like spacy without any further split or other operations
train_vocab = None 

train_vocab_path = '/home/gezi/data/kaggle/toxic/ori_vocab.txt'
MIN_COUNT = 20

SimpleTokens = namedtuple('SimpleTokens', ['tokens', 'ori_tokens', 'attributes'])
Tokens = namedtuple('Tokens', ['tokens', 
                              'attributes',
                              'ori_tokens',
                              'poses', 
                              'tags',
                              'ners',
                              ])

def init(vocab_path='/home/gezi/data/glove/glove-vocab.txt'):
  global vocab,  train_vocab
  if vocab:
    return vocab, train_vocab
  vocab = set()  
  for line in open(vocab_path, encoding='utf-8', errors='ignore'):
    vocab.add(line.rstrip('\n').split('\t')[0])

  # train_vocab = {}
  # for line in open(train_vocab_path):
  #   word, count = line.rstrip('\n').split('\t')
  #   train_vocab[word] = int(count)

  return vocab, train_vocab

# def train_dict_has(word):
#   for w in (word, word.lower(), word.capitalize(), word.upper()):
#     if w in train_vocab and train_vocab[w] > MIN_COUNT:
#       return True
#   return False

def dict_has(word):
  for w in (word, word.lower(), word.capitalize(), word.upper()):
    if w in vocab:
      return True
  return False 

def has(word):
  if not word.strip():
    return False
  #return train_dict_has(word) and dict_has(word)
  return dict_has(word)


# problem here is will also remove some other language word like '你' TODO
def en_filter(token):
  en_results = []
  results = []
  ens = []
  non_ens = []
  for x in token:
    #if x >= 'a' and x <= 'z' or x >= 'A' and x <= 'Z' or x >= '0' and x <= '9':
    if x >= 'a' and x <= 'z' or x >= 'A' and x <= 'Z':
      if non_ens:
        results.append(''.join(non_ens))
        non_ens = []
      ens.append(x)
    else:
      if ens:
        results.append(''.join(ens))
        en_results.append(results[-1])
        ens = []
      non_ens.append(x)
  if ens:
    results.append(''.join(ens))
    en_results.append(results[-1])
  if non_ens:
    results.append(''.join(non_ens))
  
  return results, en_results

# def can_split(w1, w2):
#   if train_dict_has(w1):
#     if train_dict_has(w2) or dict_has(w2):
#       return True 
#     else:
#       return False
#   else:
#     if dict_has(w1) and train_dict_has(w2):
#       return True 
#     else:
#       return False
def can_split(w1, w2):
  return dict_has(w1) and dict_has(w2) or is_toxic(w1) or is_toxic(w2)
  
def try_split(token):
  if len(token) < 6 or has(token):
    return [token]
  
  start = 3
  end = len(token) - 2
  idx = int(len(token) / 2)

  for i in range(idx, end):
    w1 = token[:i]
    w2 = token[i:]
    #print('w1:', w1, 'w2:', w2, can_split(w1, w2), train_dict_has(w1), dict_has(w1), train_dict_has(w2), dict_has(w2))
    if can_split(w1, w2):
      return [w1, '<JOIN>', w2]

  for i in reversed(range(start, idx)):
    w1 = token[:i]
    w2 = token[i:]
    #print('w1:', w1, 'w2:', w2, can_split(w1, w2), train_dict_has(w1), dict_has(w1), train_dict_has(w2), dict_has(w2))
    if can_split(w1, w2):
      return [w1, '<JOIN>', w2]
  
  return [token]

attribute_names = ['len', 'deform', 'lower', 'upper', 'has_star', 'has_dot', 'has_bracket', 'not_en']
attribute_default_values = [0.] * len(attribute_names)
Attributes = namedtuple('Attributes', attribute_names)

assert(len(attribute_names) == len(attribute_default_values))

special_tokens = set(['<N>', '<SEP>', '<JOIN>'])

# toxic_words = set([
#   'fuck', 'fucking', 'fuckin', 
#   'cunt', 'cunts',
#   'dick', 'penis', 'bitch', 'nigger', 'die', 'kill'])

def is_toxic(word):
  for w in (word, word.lower(), word.capitalize(), word.upper()):
    if w in toxic_words.get_toxic_words():
      return True
  return False

# def maybe_toxic(word):
#   for w in toxic_words:
#     if w in word:
#       return True
#   return False

def get_token_len(token):
  if token in special_tokens:
    return 1
  return len(token)


def is_en(token):
  for x in token:
    if x >= 'a' and x <= 'z' or x >= 'A' and x <= 'Z' or x >= '0' and x <= '9':
      return True 
  return False

def get_attr(token, 
             deform=False,
             has_star=False, 
             has_dot=False,
             has_bracket=False,
             not_en=False):
  return [get_token_len(token), 
          deform,
          token not in special_tokens and token.islower(), \
          token not in special_tokens and token.isupper(), \
          has_star, has_dot, has_bracket, not is_en(token)]


def tokenize(text):
  init()
  text = normalize(text)

  tokens = gezi.segment.tokenize_filter_empty(text)
  results = []
  attributes = []
  ori_tokens = []

  def append(token, ori_token, attr=None):
    results.append(token)
    ori_tokens.append(ori_token)
    attributes.append(attr or get_attr(token))

  for token in tokens:
    ori_token = token

    #print('results', results)
    if token in tokens_map:
      token = tokens_map[token]
      append(token, ori_token)
    else:
      if FLAGS.is_twitter:
        token = token.lower()
      else:
        if re.match(ip_pattern, token):
          token = '<IP>'

        # NOTICE! if http hurt perf, remove!
        if re.match(http_pattern, token):
          token = '<HTTP>'

      if has(token):
        append(token, ori_token)
      else:
        tokens, en_tokens = en_filter(token)
        tokens = [x for x in tokens if x.strip()]
        en_token = ''.join(en_tokens)
        #print('----...', tokens, en_token, en_tokens)
        # Nig(g)er -> Nigger but lose some info might just 'Nig', '<SEP>', 'g', '<SEP>', 'er' ? or mark as deformed word! TODO add to word vector
        if has(en_token):
          has_star = '*' in token
          has_dot = '.' in token
          has_bracket = '(' in token or ')' in token or '[' in token or ']' in token or '（' in token or '）' in token
          is_deform = is_toxic(en_token)

          if is_deform:
            print(en_token, token, ori_token)
          
          attr = [len(token), 
                  is_deform, en_token.islower(), en_token.isupper(), 
                  has_star, has_dot, has_bracket, False]
          if is_deform:
            append(en_token, ori_token, attr)
          else:
            append(token, ori_token, attr)
        else:
          token_results = try_split(en_token)
          if len(token_results) == 1:
            token_results = []
            for token in en_tokens:
              #print('----', token)
              token_results += try_split(token)
              token_results += ['<SEP>']
            if token_results:
              del token_results[-1]
              for token in token_results:
                append(token, ori_token, get_attr(token, True))
            else:
              append(token, ori_token)
          else:
            for token in token_results:
              append(token, ori_token, get_attr(token, True))

  if not results:
    token = 'ok'
    append(token, token)

  assert len(results) == len(attributes)
  return SimpleTokens(*([results, ori_tokens, attributes]))

# TODO merge code

    
def full_tokenize(text):
  init()
  # can cause http.. as PERSON
  text = normalize(text)
  doc = gezi.doc(text)
  results = []
  attributes = []
  poses = []
  tags = []
  ners = []
  ori_tokens = []

  def append(token, ori_token, ner='NONE', attr=None):
    results.append(token)
    poses.append(ori_token.pos_)
    tags.append(ori_token.tag_)
    attributes.append(attr or get_attr(token))
    ners.append(ner)
    ori_tokens.append(ori_token.text.replace(' ', '').replace('NEWLINE', '\x01'))
    
  ner_idx = 0
  ner_list = [(x.text, x.label_) for x in doc.ents]
  
  ner_ok = True 
  for x, y in ner_list:
    if 'NEWLINE' in x:
      ner_ok = False 
      break 
    
  #print('-----ner list', ner_list, ner_ok)
  if not ner_ok:
    ner_list = []

  for token_ in doc:
    token = token_.text
    
    # NOTICE! filtered empty text, if not filter later you must not split by  ' ', here already remove will ok
    if not token.strip():
      continue

    if FLAGS.is_twitter:
      token = token.lower()
    else:
      if re.match(ip_pattern, token):
        token = '<IP>'

    # NOTICE! if http hurt perf, remove!
    if re.match(http_pattern, token):
      token = '<HTTP>'

    # TODO better..
    ner = 'NONE'
    for i in range(ner_idx, len(ner_list)):
      if token == ner_list[i][0] or (len(token) > 2 and token in ner_list[i][0]):
        ner = ner_list[i][1]
        ner_idx = i + 1
        break
    #if ner != 'None':
    #  print(token, ner)

    if token in tokens_map:
      token = tokens_map[token]
      append(token, token_, ner)
    else:
      if FLAGS.is_twitter:
        token = token.lower()
      else:
        if re.match(ip_pattern, token):
          token = '<IP>'

        # NOTICE! if http hurt perf, remove!
        if re.match(http_pattern, token):
          token = '<HTTP>'

      #if has(token) or (ner != 'NONE' and not maybe_toxic(token)):
      if has(token):
      #if has(token) or ner == 'PERSON':
        append(token, token_, ner)
      else:
        #print('token', token)
        tokens, en_tokens = en_filter(token)
        tokens = [x for x in tokens if x.strip()]
        en_token = ''.join(en_tokens)
        #print('!!!', tokens, en_tokens, en_token)
        # Nig(g)er -> Nigger but lose some info might just 'Nig', '<SEP>', 'g', '<SEP>', 'er' ? or mark as deformed word! TODO add to word vector
        #if has(en_token) or (ner != 'NONE' and not maybe_toxic(token)):
        if has(en_token):
        #if has(en_token) or ner == 'PERSON':
          has_star = '*' in token
          has_dot = '.' in token
          has_bracket = '(' in token or ')' in token or '[' in token or ']' in token or '（' in token or '）' in token
          is_deform = is_toxic(en_token)
          attr = [len(token), is_deform,
                  en_token.islower(), en_token.isupper(), 
                  has_star, has_dot, has_bracket, False]
          if is_deform:
            append(en_token, token_, ner, attr)
          else:
            append(token, token_, ner, attr)
        else:
          token_results = try_split(en_token)
          if len(token_results) == 1:
            token_results = []
            for token in en_tokens:
              #print('----', token)
              token_results += try_split(token)
              token_results += ['<SEP>']
            if token_results:
              del token_results[-1]
              for token in token_results:
                append(token, token_, ner, get_attr(token, True))
            else:
              append(token, token_, ner)
          else:
            for token in token_results:
              append(token, token_, ner, get_attr(token, True))

  if not results:
    return full_tokenize('ok')
  assert len(results) == len(attributes) == len(ori_tokens) == len(poses) == len(tags) == len(ners)
  return Tokens(*([results, attributes, ori_tokens, poses, tags, ners]))
