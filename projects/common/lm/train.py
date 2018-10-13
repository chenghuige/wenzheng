#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2018-01-13 16:32:26.966279
#   \Description  
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_bool('concat_layers', False, '')

import numpy as np
from tqdm import tqdm

import melt 
logging = melt.logging
import gezi
import traceback


from algos.loss import loss_fn
#import algos.model as base
from algos.model import PTBModel as Model
from dataset import Dataset
#import evaluate as ev

# EAGER=1 python train.py 
# EAGER=1 python train.py --word_embedding_file ./mount/temp/ai2018/sentiment/tfrecords/char.glove/emb.npy 

def main(_):
  vocab = gezi.Vocabulary(FLAGS.vocab)

  # FLAGS.valid_input = './mount/temp/lm/corpus/sentiment/valid/ids.npy'
  # FLAGS.train_input = './mount/temp/lm/corpus/sentiment/train/ids.npy'

  # FLAGS.model_dir = './mount/temp/lm/model/sentiment' if not FLAGS.word_embedding_file else './mount/temp/lm/model/sentiment.pretrain'

  # FLAGS.batch_size_dim = 0

  melt.apps.init()

  #ev.init()

  #model = getattr(base, FLAGS.model)()

  model = Model(vocab_size=vocab.size(),
                embedding_dim=FLAGS.emb_dim,
                hidden_dim=FLAGS.rnn_hidden_size,
                num_layers=FLAGS.num_layers,
                concat_layers=FLAGS.concat_layers,
                dropout_ratio=1 - FLAGS.keep_prob)

  logging.info(model)

  train = melt.apps.get_train()

  train(Dataset,
        model,  
        loss_fn)   

if __name__ == '__main__':
  tf.app.run()  
