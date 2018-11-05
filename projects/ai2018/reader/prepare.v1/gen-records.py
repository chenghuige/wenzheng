#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-records.py
#        \author   chenghuige  
#          \date   2018-08-29 15:20:35.282947
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', './mount/data/ai2018/reader/valid.json', '') 
flags.DEFINE_string('vocab_', './mount/temp/ai2018/reader/tfrecord/vocab.txt', 'vocabulary txt file')
#flags.DEFINE_string('seg_method', 'basic', '') 
flags.DEFINE_bool('binary', False, '')
flags.DEFINE_integer('limit', 5000, '')
flags.DEFINE_integer('max_examples', None, '')
flags.DEFINE_integer('threads', None, '')
flags.DEFINE_bool('use_char', False, '')

import traceback
import multiprocessing
from sklearn.utils import shuffle
import numpy as np
import glob
import json

from gezi import Vocabulary
import gezi
import melt
from text2ids import text2ids as text2ids_ 
from wenzheng.utils import text2ids

from multiprocessing import Value, Manager
counter = Value('i', 0) 
total_words = Value('i', 0)

def get_mode(path):
  if 'train' in path:
    return 'train'
  elif 'valid' in path:
    return 'valid' 
  elif 'test' in path:
    return 'test'
  elif '.pm' in path:
    return 'pm'
  return 'train'


def is_negative(candidate):
  negs = ['不', '无', '没']
  for neg in negs:
    if neg in candidate:
      return True 
  return False

# neg, pos, uncertain
def sort_alternatives(alternatives, query):
  candidates = [None] * 3
  type = 0

  l = []
  alternatives = alternatives.split('|')
  # TODO check strip.. for answer
  for candidate_ in alternatives:
    # if candidate != candidate.strip():
    #   print(query, alternatives)
    candidate = candidate_.strip()
    if candidate == '无法确定':
      candidates[2] = candidate_
    else:
      l.append(candidate_) 

  if candidates[2] == None:
    l = []
    for candidate_ in alternatives:
      candidate = candidate_.strip()
      if ('无法' in candidate or '确定' in candidate or '确认' in candidate) or candidate == 'wfqd':
        candidates[2] = candidate_
      else:
        l.append(candidate_) 

  if candidates[2] == None:
    candidates[2] = l[-1]

  if len(l) == 0:
    candidates[0] = candidates[2]
    candidates[1] = candidates[2]
    
    return candidates, type

  # TODO "alternatives": "不能|无法确定" if only 2 possbiles now make another as 无法确定 but might just set to '' and
  # when inference only consider prob of 不能 and 无法确定 mask '' to 0 prob
  if len(l) == 1:
    if l[0].strip().startswith('不') or l[0].strip().startswith('无') \
      or l[0].strip().startswith('假') \
      or l[0].strip().startswith('坏'):
      candidates[0] = l[0]
      candidates[1] = candidates[-1]
    else:
      candidates[0] = candidates[-1]
      candidates[1] = l[0]

    return candidates, type

  if l[0].strip() == '是' and l[1].strip() == '否':
    candidates[0] = l[1]
    candidates[1] = l[0]
    return candidates, type
  
  if l[0].strip() == '否' and l[1].strip() == '是':
    candidates[0] = l[0]
    candidates[1] = l[1]
    return candidates, type

  if l[0].strip().startswith('真') and l[1].strip().startswith('假') or l[0].strip().startswith('好') and l[1].strip().startswith('坏'):
    candidates[0] = l[1]
    candidates[1] = l[0]
    return candidates, type

  if l[0].strip().startswith('假') and l[1].strip().startswith('真') or l[0].strip().startswith('坏') and l[1].strip().startswith('好'):
    candidates[0] = l[0]
    candidates[1] = l[1]
    return candidates, type

  if l[0] in l[1]:
    candidates[0] = l[1]
    candidates[1] = l[0]
    return candidates, type
  
  if l[1] in l[0]:
    candidates[0] = l[0]
    candidates[1] = l[1]
    return candidates, type    


  if is_negative(l[0]):
    candidates[0] = l[0]
    candidates[1] = l[1]
    return candidates, type
  elif is_negative(l[1]):
    candidates[0] = l[1]
    candidates[1] = l[0]
    return candidates, type
  
  if l[0] in query and l[1] not in query:
    candidates[1] = l[0]
    candidates[0] = l[1]
    return candidates, type
  
  if l[0] not in query and l[1] in query:
    candidates[0] = l[0]
    candidates[1] = l[1]
    return candidates, type
      
  # #assert l[0] in query and l[1] in query, f'{l[0]},{l[1]},{query}' 
  # if not (l[0] in query and l[1] in query):
  #   return [], type 

  type = 1
  # TODO not ok for like  跑步好，慢走好  跑步和慢走哪个好？
  pos1 = query.lower().find(l[0].strip().lower())
  if pos1 < 0:
    pos1 = query.lower().find(l[0][:-1].strip().lower())
  pos2 = query.lower().find(l[1].strip().lower())
  if pos2 < 0:
    pos2 = query.lower().find(l[1][:-1].strip().lower())
  if pos1 < 0:
    pos1 = 10000
  if pos2 < 0:
    pos2 = 10000
  if pos1 <= pos2:
    # candidates[0] = l[0]
    # candidates[1] = l[1]
    candidates[0] = l[1]
    candidates[1] = l[0]
  else:
    # candidates[0] = l[1]
    # candidates[1] = l[0]
    candidates[0] = l[0]
    candidates[1] = l[1]

  return candidates, type

def build_features(file_):
  mode = get_mode(FLAGS.input)
  out_file = os.path.dirname(FLAGS.vocab_) + '/{0}/{1}_{2}.tfrecord'.format(mode, os.path.basename(os.path.dirname(file_)), os.path.basename(file_))
  os.system('mkdir -p %s' % os.path.dirname(out_file))
  print('infile', file_, 'out_file', out_file)

  num = 0
  num_whether = 0
  answer_len = 0
  with melt.tfrecords.Writer(out_file) as writer:
    for line in open(file_):
      try:
        m = json.loads(line.rstrip('\n'))
        url = m['url']
        alternatives = m['alternatives']
        query_id = int(m['query_id'])
        passage = m['passage']
        query = m['query']

        # if query_id != 254146:
        #   continue

        if not 'answer' in m:
          answer = 'unknown'
        else:
          answer = m['answer']

        # candidates is neg,pos,uncertain
        # type 0 means true or false,  type 1 means wehter
        candidates, type = sort_alternatives(alternatives, query)

        assert candidates is not None

        answer_id = 0
        for i, candiate in enumerate(candidates):
          if candiate == answer:
            answer_id = i

        assert candidates is not None
        candidates_str = '|'.join(candidates)

        query_ids = text2ids_(query)
        passage_ids = text2ids_(passage)

        candidate_neg_ids = text2ids_(candidates[0])
        candidate_pos_ids = text2ids_(candidates[1])
        candidate_na_ids = text2ids_('无法确定')

        if len(candidate_pos_ids) > answer_len:
          answer_len = len(candidate_pos_ids)
          print(answer_len)
        if len(candidate_neg_ids) > answer_len:
          answer_len = len(candidate_neg_ids)
          print(answer_len)
        
        assert len(query_ids), line
        assert len(passage_ids), line

        limit = FLAGS.limit

        if len(passage_ids) > limit:
          print('long line', len(passage_ids), query_id)

        query_ids = query_ids[:limit]
        passage_ids = passage_ids[:limit]

        feature = {
                    'id': melt.bytes_feature(str(query_id)),
                    'url': melt.bytes_feature(url),
                    'alternatives': melt.bytes_feature(alternatives),
                    'candidates': melt.bytes_feature(candidates_str),
                    'passage': melt.int64_feature(passage_ids),
                    'passage_str': melt.bytes_feature(passage),
                    'query': melt.int64_feature(query_ids),
                    'query_str': melt.bytes_feature(query),
                    'candidate_neg': melt.int64_feature(candidate_neg_ids),
                    'candidate_pos': melt.int64_feature(candidate_pos_ids),
                    'candidate_na': melt.int64_feature(candidate_na_ids),
                    'answer': melt.int64_feature(answer_id),
                    'answer_str': melt.bytes_feature(answer),
                    'type': melt.int64_feature(type)         
                  }

        # TODO currenlty not get exact info wether show 1 image or 3 ...
        record = tf.train.Example(features=tf.train.Features(feature=feature))

        #if not candidates:
        if num % 1000 == 0:
          print(num, query_id, query, type)
          print(alternatives, candidates)
          print(answer, answer_id)

        writer.write(record)
        num += 1
        if type:
          num_whether += 1
        global counter
        with counter.get_lock():
          counter.value += 1
        global total_words
        with total_words.get_lock():
          total_words.value += len(passage_ids)
        if FLAGS.max_examples and num >= FLAGS.max_examples:
          break
      except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        print('-----------', query)
        print(alternatives)

      #break
  print('num_wehter:', num_whether)

def main(_):  
  text2ids.init(FLAGS.vocab_)
  print('to_lower:', FLAGS.to_lower, 'feed_single_en:', FLAGS.feed_single_en, 'seg_method', FLAGS.seg_method)
  print(text2ids.ids2text(text2ids_('傻逼脑残B')))
  print(text2ids_('傻逼脑残B'))
  print(text2ids.ids2text(text2ids_('喜欢玩孙尚香的加我好友：2948291976')))

  #exit(0)
  
  if os.path.isfile(FLAGS.input):
    build_features(FLAGS.input)
  else:
    files = glob.glob(FLAGS.input + '/*') 
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(build_features, files)
    pool.close()
    pool.join()


  # for safe some machine might not use cpu count as default ...
  print('num_records:', counter.value)
  mode = get_mode(FLAGS.input)

  os.system('mkdir -p %s/%s' % (os.path.dirname(FLAGS.vocab_), mode))
  out_file = os.path.dirname(FLAGS.vocab_) + '/{0}/num_records.txt'.format(mode)
  gezi.write_to_txt(counter.value, out_file)

  print('mean words:', total_words.value / counter.value)

if __name__ == '__main__':
  tf.app.run()
