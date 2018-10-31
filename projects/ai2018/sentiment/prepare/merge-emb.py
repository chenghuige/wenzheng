#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   merge-glove.py
#        \author   chenghuige  
#          \date   2018-01-15 23:52:08.616633
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('input_vocab', './mount/temp/ai2018/sentiment/tfrecord/vocab.ori.txt', '')
flags.DEFINE_string('emb', './mount/temp/ai2018/sentiment/vectors.txt', '')
flags.DEFINE_integer('min_count', 20, '')
flags.DEFINE_integer('most_common', None, '')
flags.DEFINE_integer('emb_dim', 300, '')
flags.DEFINE_string('out_name', 'emb.npy', '')
flags.DEFINE_string('type', 'normal', '''normal try merge all in glove, and add not in glove ones which with min train count,
                                         scratch add all with min train count and try merge glove,
                                         only only merge all glove and not consider train count''')

flags.DEFINE_integer('max_words', 200000, '')
flags.DEFINE_bool('add_additional', True, 'add additional words from embedding file, this is by default for word')

flags.DEFINE_string('sort_by', 'count', '')

from tqdm import tqdm
import numpy as np

def main(_):
  print('emb', FLAGS.emb)
  input_vocab = FLAGS.input_vocab
  print('input_vocab', input_vocab)
  dir_ = os.path.dirname(FLAGS.input_vocab)
  lines = open(input_vocab).readlines()

  ori_words_counts = [x.rstrip('\n').split('\t') for x in lines]
  # TODO FIXME why must remove? other wise when in  for word, count in zip(ori_words, counts): will ValueError: invalid literal for int() with base 10: '    '
  ori_words_counts = filter(lambda x: x[0].strip(), ori_words_counts)
  ori_words, counts = zip(*ori_words_counts)
  counts = list(map(int, counts))
  ori_set = set(ori_words)

  normed_ori_set = set([x.lower() for x in ori_set])
  
  embedding_dict = {}

  vec_size = FLAGS.emb_dim
  #with open(FLAGS.emb, 'r') as fh:
  with open(FLAGS.emb, 'r', encoding='utf-8', errors='ignore') as fh:
    #for line in tqdm(fh, total=2196017):
    for i, line in enumerate(fh):
      array = line.split()
      # fasttext txt has header line
      if len(array) < vec_size:
        continue 
      word = "".join(array[0:-vec_size])
      try:
        vector = list(map(float, array[-vec_size:]))
      except Exception:
        print(i, line)
        continue
      if word.lower() in normed_ori_set:
        embedding_dict[word] = vector
      else:
        embedding_dict[word.lower()] = vector
      if i % 100000 == 0:
        print(i)
        #break
  print("{} / {} tokens have corresponding word embedding vector".format(len(embedding_dict), len(ori_words)))
  
  words = []   
  emb_mat = []
  # for padding zero
  emb_mat.append(np.array([0.] * vec_size))
  if not '<UNK>' in ori_set:
    #change from all 0 to random normal for unk
    #emb_mat.append([np.random.normal(scale=0.1) for _ in range(vec_size)])
    emb_mat.append([np.random.uniform(-0.08, 0.08) for _ in range(vec_size)])
    words.append('<UNK>')

  if FLAGS.sort_by == 'count':
    print('sort by count')
    with open('/home/gezi/tmp/rare_words.txt', 'w') as rare_out:
      for word, count in zip(ori_words, counts):
        #if FLAGS.type == 'normal':
        if word in embedding_dict:
          emb_mat.append(np.array(embedding_dict[word]))
          words.append(word)  
        else:
          if count >= FLAGS.min_count:
            print('%s %d' % (word, count), file=rare_out)
            #emb_mat.append([np.random.normal(scale=0.1) for _ in range(vec_size)])
            #emb_mat.append([np.random.uniform(-0.08, 0.08) for _ in range(vec_size)])
            emb_mat.append([np.random.uniform(-0.08, 0.08) for _ in range(vec_size)])
            words.append(word)  
  else:
    print('sort by knowldege')
    words_knowledge = []
    emb_mat_knowledge = []
    words_no_knowledge = []
    emb_mat_no_knowledge = []
    with open('/home/gezi/tmp/rare_words.txt', 'w') as rare_out:
      for word, count in zip(ori_words, counts):
        #if FLAGS.type == 'normal':
        if word in embedding_dict:
          emb_mat_knowledge.append(np.array(embedding_dict[word]))
          words_knowledge.append(word)  
        else:
          if count >= FLAGS.min_count:
            print('%s %d' % (word, count), file=rare_out)
            #emb_mat.append([np.random.normal(scale=0.1) for _ in range(vec_size)])
            #emb_mat.append([np.random.uniform(-0.08, 0.08) for _ in range(vec_size)])
            emb_mat_no_knowledge.append([np.random.uniform(-0.08, 0.08) for _ in range(vec_size)])
            words_no_knowledge.append(word)  
        # elif FLAGS.type == 'scratch':
        #   if count >= FLAGS.min_count:
        #     if word in embedding_dict:
        #       emb_mat.append(np.array(embedding_dict[word]))
        #       words.append(word)  
        #     else:
        #       #emb_mat.append([np.random.normal(scale=0.1) for _ in range(vec_size)])
        #       emb_mat.append([np.random.uniform(-0.08, 0.08) for _ in range(vec_size)])
        #       words.append(word)  
        # elif FLAGS.type == 'only':
        #   if word in embedding_dict:
        #     emb_mat.append(np.array(embedding_dict[word]))
        #     words.append(word)  

    #if FLAGS.sort_by == 'knowledge':
    words += words_no_knowledge
    emb_mat += emb_mat_no_knowledge
    words += words_knowledge
    emb_mat += emb_mat_knowledge

  words_set = set(words)

  unk_words = []

  # added from v11 to let less unk
  for word, count in zip(ori_words, counts):
    if word not in words_set:
      contains = False
      for w in (word.lower(), word.capitalize(), word.upper()):
        if w in words_set:
          contains = True
      if not contains:
        for w in (word.lower(), word.capitalize(), word.upper()):
          if w in embedding_dict:
            #print('adding....', w, word)
            words_set.add(w)
            emb_mat.append(np.array(embedding_dict[w]))
            words.append(w)
            contains = True
            break
      if not contains:
        unk_words.append(word)
        #emb_mat.append([np.random.uniform(-0.08, 0.08) for _ in range(vec_size)])

  # emb_mat is with all word (train + test)
  print('num in vocab words', len(words))
  print('num oov words', len(unk_words))

  if FLAGS.add_additional:
    for word in embedding_dict:
      if word not in words_set:
        words_set.add(word)
        words.append(word)
        emb_mat.append(np.array(embedding_dict[word]))
        if len(words) > FLAGS.max_words:
          break

    print('num words after adding additional', len(words))

  out_vocab = os.path.join(dir_, 'vocab.txt')
  # NOTICE unk vocab actually means only top words, so for train we can use <UNK>
  # out_unk_vocab = os.path.join(FLAGS.dir, 'unk_vocab.txt')
  # with open(out_vocab, 'w') as out, open(out_unk_vocab, 'w') as out_unk:
  #   for word in words:
  #     print(word, file=out)
  #     print(word, file=out_unk)
  #   for word in unk_words:
  #     print(word, file=out) 
  print('out_vocab', out_vocab)
  with open(out_vocab, 'w') as out:
    for word in words:
      print(word, file=out)

  out_mat = os.path.join(dir_, FLAGS.out_name)
  print('len(emb_mat)', len(emb_mat))
  print('out_mat', out_mat)
  np.save(out_mat, np.array(emb_mat))
    
if __name__ == '__main__':
  tf.app.run()
