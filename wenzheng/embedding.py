#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   embedding.py
#        \author   chenghuige  
#          \date   2016-12-24 19:55:37.327855
#   \Description  
# ==============================================================================
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
#import gflags as flags
FLAGS = flags.FLAGS
  
flags.DEFINE_integer('emb_dim', 512, 'embedding dim for each word, notice for rnn bidirectional here should be acutal emb_dim * 2')
flags.DEFINE_integer('char_emb_dim', 300, 'embedding dim for each word, notice for rnn bidirectional here should be acutal emb_dim * 2')
flags.DEFINE_float('weight_stddev', 1e-4,  
                                  """weight stddev, 
                                     @Notice if use bias then small stddev like 0.01 might not lead to convergence, 
                                     causing layer weight value always be 0 with random_normal""")
flags.DEFINE_float('initializer_scale', 0.08, 'used for weights initalize using random_uniform, default value 0.08 follow im2txt')

flags.DEFINE_string('word_embedding_file', None, 'load pre trained word embedding from word2vec or glov if not None')
flags.DEFINE_boolean('finetune_word_embedding', True, 'wether update word embedding')

flags.DEFINE_string('char_embedding_file', None, 'load pre trained char embedding from word2vec or glov if not None')
flags.DEFINE_boolean('finetune_char_embedding', True, 'wether update char embedding')

flags.DEFINE_string('pinyin_embedding_file', None, 'load pre trained pinyin embedding from word2vec or glov if not None')
flags.DEFINE_boolean('finetune_pinyin_embedding', True, 'wether update pinyin embedding')

flags.DEFINE_string('ngram_embedding_file', None, 'load pre trained ngram embedding from word2vec or glov if not None')
flags.DEFINE_boolean('finetune_ngram_embedding', True, 'wether update ngram embedding')

flags.DEFINE_boolean('position_embedding', False, 'wether use postion embedding')

flags.DEFINE_string('ngram_vocab', None, '')
flags.DEFINE_string('char_vocab', None, '')
flags.DEFINE_string('pinyin_vocab', None, '')

flags.DEFINE_string('emb_init', 'uniform', 'uniform or normal normal with hidden_size ** -0.5 might perform better')
flags.DEFINE_float('emb_stddev', 0., 'if 0 use hidden_size ** -0.5, another choice might be set to 0.01 as HKUST rnet')


#import tensorflow.contrib.slim as slim

import sys
import os

import numpy as np

import melt
logging = melt.logging
import gezi

from wenzheng.utils import vocabulary  

import glob

try:
  import conf 
  from conf import TEXT_MAX_WORDS
except Exception:
  print('Warning, no conf.py in current path use util conf', file=sys.stderr)
  from wenzheng.utils.conf import TEXT_MAX_WORDS


layers = tf.keras.layers

class Embedding(layers.Layer):
  """An Embedding layer."""
  
  def __init__(self, vocab_size, embedding_dim=None, embedding=None, 
               trainable=True, freeze_size=None, finetune_size=None, **kwargs):
    super(Embedding, self).__init__(**kwargs)
    self.vocab_size = vocab_size
    self.freeze_size = freeze_size
    if finetune_size:
      self.freeze_size = vocab_size - finetune_size
    self.embedding_dim = embedding_dim if embedding_dim else FLAGS.emb_dim
    self.trainable = trainable

    self.embedding = embedding
    self.embedding2 = None
    if embedding is not None:
      if type(embedding) is str:
        if os.path.exists(embedding):
          embedding = np.load(embedding)
        else:
          embedding = None
      self.embedding = embedding
      
  def build(self, _):
    initializer = 'uniform'
    # some optimizer must use embedding on cpu 
    #with tf.device("/cpu:0"):
    if self.embedding is not None:
      initializer = tf.constant_initializer(self.embedding)
      logging.info('emb init from numpy pretrain and trainable:', self.trainable)
    else:
      if FLAGS.emb_init == 'uniform':
        init_width = 0.5 / self.embedding_dim
        logging.info('emb random_uniform init with width:', init_width)
        initializer = tf.random_uniform_initializer(-init_width, init_width)
      elif FLAGS.emb_init == 'normal' or FLAGS.emb_init == 'random':
        stddev = FLAGS.emb_stddev or self.embedding_dim ** -0.5
        logging.info('emb random_normal init with stddev:', stddev)
        initializer = tf.random_normal_initializer(mean=0., stddev=stddev)

    self.embedding = self.add_variable(
        "embedding_kernel",
        shape=[int(self.vocab_size), self.embedding_dim],
        dtype=tf.float32,
        initializer=initializer,
        trainable=self.trainable)

    if self.freeze_size:
      assert self.trainable
      embedding, embedding2 = tf.split(self.embedding, [self.vocab_size - self.freeze_size, self.freeze_size], 0)
      self.embedding = tf.concat([embedding, tf.stop_gradient(embedding2)], 0)

  def call(self, x):
    #print('---------', self.embedding)
    return tf.nn.embedding_lookup(self.embedding, x)

#TODO try l2_regularizer and compare
#weights = slim.variable('weights',
#                             shape=[10, 10, 3 , 3],
#                             initializer=tf.truncated_normal_initializer(stddev=0.1),
#                             regularizer=slim.l2_regularizer(0.05),
#                             device='/CPU:0')
def get_embedding(name='emb', height=None, emb_dim=None, trainable=True):
  emb_dim = emb_dim or FLAGS.emb_dim
  if height is None:
    vocabulary.init()
    height = vocabulary.get_vocab_size() 
  
  # google transform use below
  #initializer=tf.random_normal_initializer(
  #            0., self.hidden_size ** -0.5)
  # squad use np.random.normal(scale=0.01)

  if FLAGS.emb_init == 'uniform':
    init_width = 0.5 / emb_dim
    emb = melt.variable.get_weights_uniform(name, [height, emb_dim], -init_width, init_width, trainable=trainable)
    logging.info('emb random_uniform init with width', init_width)
  elif FLAGS.emb_init == 'normal' or FLAGS.emb_init == 'random':
    stddev = FLAGS.emb_stddev or emb_dim ** -0.5
    logging.info('emb random_normal init with stddev', stddev)
    emb = melt.variable.get_weights_random(name, [height, emb_dim], stddev, trainable=trainable)
  else:
    raise ValueError(FLAGS.emb_init)

  #return to above code if this works not better
  #emb = melt.variable.get_weights_truncated(name, [vocab_size, emb_dim], stddev=FLAGS.weight_stddev)
  return emb 

def get_embedding_cpu(name='emb', height=None, emb_dim=None, trainable=True):
  with tf.device('/CPU:0'):
    return get_embedding(name, height=height, emb_dim=emb_dim, trainable=trainable)

def get_or_restore_embedding(name='emb', embedding_file=None, trainable=None, height=None, emb_dim=None, type='word'):
  # cpu for adgrad optimizer
  #if (not FLAGS.word_embedding_file) or glob.glob(FLAGS.model_dir + '/model*ckpt*'):
    # logging.info('Word embedding random init or from model_dir:{} and trainable=:{}'.format(
    #     FLAGS.model_dir, FLAGS.finetune_word_embedding))
  #TODO verify below is ok , above is ok but a bit complex. I assume if var in check point will later restore and cover initital const value
  #if not FLAGS.word_embedding_file:
  embedding_file_ = None 
  train_able_ = None
  if type == 'word':
    embedding_file_ = FLAGS.word_embedding_file
    train_able_ = FLAGS.finetune_word_embedding
  elif type == 'char':
    embedding_file_ = FLAGS.char_embedding_file
    train_able_ = FLAGS.finetune_char_embedding
  elif type == 'ngram':
    embedding_file_ = FLAGS.ngram_embedding_file
    train_able_ = FLAGS.finetune_ngram_embedding   
  elif type == 'pinyin':
    embedding_file_ = FLAGS.pinyin_embedding_file
    train_able_ = FLAGS.finetune_pinyin_embedding       
  else:
    raise ValueError(type)
    
  embedding_file = embedding_file if embedding_file is not None else embedding_file_
  trainable = trainable if trainable is not None else train_able_

  #logging.info('----------------------', type, embedding_file, height)
  if (not embedding_file) or melt.exists_model(FLAGS.model_dir):
    logging.info('{} random init or from model_dir and trainable=:{}'.format(name, trainable))
    emb = get_embedding(
        name=name, trainable=trainable, height=height, emb_dim=emb_dim)
    #melt.try_add_to_collection('word_embedding', emb)
  else:
    # https://github.com/tensorflow/tensorflow/issues/1570
    # still adgrad must cpu..
    # if not fintue emb this will be ok if fintune restart will ok ? must not use word embedding file? os.path.exists(FLAGS.model_dir) ? judge?
    # or will still try to load from check point ? TODO for safe you could re run by setting word_embedding_file as None or ''
    logging.info('Loading {} from:{} and trainable=:{}'.format(
        name, embedding_file, trainable))
    timer = gezi.Timer('load constat')
    emb = melt.load_constant(
        embedding_file, name=name, trainable=trainable)
    timer.print_elapsed()
  return emb

def get_or_restore_embedding_cpu(name='emb', embedding_file=None, trainable=None, height=None, emb_dim=None):
  with tf.device('/CPU:0'):
    return get_or_restore_embedding(name, embedding_file, trainable, height, emb_dim)

def get_position_embedding(name='pos_emb', height=None):
  if FLAGS.position_embedding:
    logging.info('Using position embedding')
    pos_emb = get_embedding(name, height=height or TEXT_MAX_WORDS)
  else:
    pos_emb = None
  return pos_emb

def get_position_embedding_cpu(name='pos_emb', height=None):
  with tf.device('/CPU:0'):
    return get_position_embedding(name, height=height or TEXT_MAX_WORDS)

#TODO height
def get_or_restore_char_embedding_cpu(name='char_emb', embedding_file=None, trainable=None):
  with tf.device('/CPU:0'):
    return get_or_restore_char_embedding(name, embedding_file, trainable)

def get_or_restore_char_embedding(name='char_emb', embedding_file=None, trainable=None):
  embedding_file = embedding_file or FLAGS.char_embedding_file 
  #assert embedding_file
  trainable = trainable or FLAGS.finetune_char_embedding
  char_vocab = FLAGS.char_vocab or FLAGS.vocab.replace('vocab.txt', 'char_vocab.txt')
  height = gezi.Vocabulary(char_vocab).size()
  assert height
  return get_or_restore_embedding(name, embedding_file, trainable, height=height, emb_dim=FLAGS.char_emb_dim, type='char')

def get_or_restore_ngram_embedding_cpu(name='ngram_emb', embedding_file=None, trainable=None):
  with tf.device('/CPU:0'):
    return get_or_restore_ngram_embedding(name, embedding_file, trainable)

def get_or_restore_ngram_embedding(name='ngram_emb', embedding_file=None, trainable=None):
  embedding_file = embedding_file or FLAGS.ngram_embedding_file 
  #assert embedding_file
  trainable = trainable or FLAGS.finetune_ngram_embedding
  ngram_vocab = FLAGS.ngram_vocab or FLAGS.vocab.replace('vocab.txt', 'ngram_vocab.txt')
  height = gezi.Vocabulary(ngram_vocab).size()
  assert height
  return get_or_restore_embedding(name, embedding_file, trainable, height=height, emb_dim=FLAGS.ngram_emb_dim, type='ngram')

def get_or_restore_pinyin_embedding_cpu(name='pinyin_emb', embedding_file=None, trainable=None):
  with tf.device('/CPU:0'):
    return get_or_restore_pinyin_embedding(name, embedding_file, trainable)

def get_or_restore_pinyin_embedding(name='pinyin_emb', embedding_file=None, trainable=None):
  embedding_file = embedding_file or FLAGS.pinyin_embedding_file 
  #assert embedding_file
  trainable = trainable or FLAGS.finetune_pinyin_embedding
  pinyin_vocab = FLAGS.pinyin_vocab or FLAGS.vocab.replace('vocab.txt', 'pinyin_vocab.txt')
  height = gezi.Vocabulary(pinyin_vocab).size()
  assert height
  return get_or_restore_embedding(name, embedding_file, trainable, height=height, emb_dim=FLAGS.ngram_emb_dim, type='pinyin')

