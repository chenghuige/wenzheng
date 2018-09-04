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

flags.DEFINE_string('dir', './mount/temp/toxic/tfrecords/glove/', '')
flags.DEFINE_string('emb', './mount/data/kaggle/toxic/talk_corpus/fastText/result.3gram.5epoch/toxic.input', '')
flags.DEFINE_integer('min_count', 10, '')
flags.DEFINE_integer('emb_dim', 300, '')
flags.DEFINE_string('out_name', 'emb.npy', '')
flags.DEFINE_string('type', 'normal', '''normal try merge all in glove, and add not in glove ones which with min train count,
                                       scratch add all with min train count and try merge glove,
                                       only only merge all glove and not consider train count''')

from tqdm import tqdm
import numpy as np
import gezi
from gezi import Vocabulary
import config

def main(_):
  input_vocab = os.path.join(FLAGS.dir, 'vocab.full.txt')
  ft_vocab = Vocabulary(os.path.join(os.path.dirname(FLAGS.emb), 'vocab.txt'), fixed=True)
  lines = open(input_vocab).readlines()

  ori_words_counts = [x.rstrip('\n').split('\t') for x in lines]
  # TODO FIXME why must remove? other wise when in  for word, count in zip(ori_words, counts): will ValueError: invalid literal for int() with base 10: '    '
  ori_words_counts = filter(lambda x: x[0].strip(), ori_words_counts)
  ori_words, counts = zip(*ori_words_counts)
  counts = list(map(int, counts))
  ori_set = set(ori_words)

  normed_ori_set = set([x.lower() for x in ori_set])
  
  embedding_dict = {}

  ngrams = []

  vec_size = FLAGS.emb_dim
  with open(FLAGS.emb, 'r', encoding='utf-8') as fh:
    #for line in tqdm(fh, total=2196017):
    for i, line in enumerate(fh):
      array = line.split()
      # fasttext txt has header line
      if len(array) < vec_size:
        continue 
      vector = list(map(float, array))
      if i >= ft_vocab.size():
        ngrams.append(vector)
        continue 
      word = ft_vocab.key(i) 
      if word.lower() in normed_ori_set:
        embedding_dict[word] = vector
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
  if not '<S>' in ori_set:
    emb_mat.append([np.random.uniform(-0.08, 0.08) for _ in range(vec_size)])
    words.append('<S>')
  if not '</S>' in ori_set:
    emb_mat.append([np.random.uniform(-0.08, 0.08) for _ in range(vec_size)])
    words.append('</S>')

  with open('/home/gezi/tmp/rare_words.txt', 'w') as rare_out:
    for word, count in zip(ori_words, counts):
      if FLAGS.type == 'normal':
        if word in embedding_dict:
          emb_mat.append(np.array(embedding_dict[word]))
          words.append(word)  
        else:
          if count >= FLAGS.min_count:
            print('%s %d' % (word, count), file=rare_out)
            #emb_mat.append([np.random.normal(scale=0.1) for _ in range(vec_size)])
            emb_mat.append([np.random.uniform(-0.08, 0.08) for _ in range(vec_size)])
            words.append(word)  
      elif FLAGS.type == 'scratch':
        if count >= FLAGS.min_count:
          if word in embedding_dict:
            emb_mat.append(np.array(embedding_dict[word]))
            words.append(word)  
          else:
            #emb_mat.append([np.random.normal(scale=0.1) for _ in range(vec_size)])
            emb_mat.append([np.random.uniform(-0.08, 0.08) for _ in range(vec_size)])
            words.append(word)  
      elif FLAGS.type == 'only':
        if word in embedding_dict:
          emb_mat.append(np.array(embedding_dict[word]))
          words.append(word)  

  words_set = set(words)

  for word, count in zip(ori_words, counts):
    if word not in words_set:
      contains = False
      for w in (word.lower(), word.capitalize(), word.upper()):
        if w in words_set:
          contains = True
      if not contains:
        for w in (word.lower(), word.capitalize(), word.upper()):
          if w in embedding_dict:
            print('adding....', w, word)
            words_set.add(w)
            emb_mat.append(np.array(embedding_dict[w]))
            words.append(w)
            break

  out_vocab = os.path.join(FLAGS.dir, 'vocab.txt')
  print('out vocab size', len(words), 'ori ft vocab size', ft_vocab.size())
  with open(out_vocab, 'w') as out:
    for word in words:
      print(word, file=out)

  out_mat = os.path.join(FLAGS.dir, FLAGS.out_name)

  emb_mat += ngrams

  # # check
  # ids = gezi.fasttext_ids('you', Vocabulary(out_vocab), FLAGS.ngram_buckets, 3, 3)
  # print('---------ids', ids)
  # vectors = []
  # for id in ids:
  #   vectors.append(emb_mat[id])
  # vectors = np.stack(vectors)
  # print(np.mean(vectors, 0))


  print('len(emb_mat)', len(emb_mat))
  np.save(out_mat, np.array(emb_mat))
    
if __name__ == '__main__':
  tf.app.run()
