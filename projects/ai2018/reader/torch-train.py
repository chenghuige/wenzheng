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

import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

import numpy as np
from tqdm import tqdm

import melt 
logging = melt.logging
import gezi
import traceback

from wenzheng.utils import input_flags 

import torch_algos.model as base
from torch_algos.loss import criterion
from dataset import Dataset
import evaluate as ev
import lele

def get_num_finetune_words():
  if not FLAGS.dynamic_finetune:
    return FLAGS.num_finetune_words
  else:
    #return min(int(melt.epoch() * 1000), FLAGS.num_finetune_words)
    return int(melt.epoch() * 1000)

def freeze_embedding(self, grad_input, grad_output):
  num_finetune_words = get_num_finetune_words()
  grad_output[0][num_finetune_words:, :] = 0

def freeze_char_embedding(self, grad_input, grad_output):
  grad_output[0][FLAGS.num_finetune_chars:, :] = 0

def main(_):
  FLAGS.num_folds = 10
  FLAGS.torch = True
  melt.apps.init()

  ev.init()

  embedding = None
  if FLAGS.word_embedding_file and os.path.exists(FLAGS.word_embedding_file):
    embedding = np.load(FLAGS.word_embedding_file)
    FLAGS.emb_dim = embedding.shape[1]

  model = getattr(base, FLAGS.model)(embedding)
  if FLAGS.num_finetune_words:
    model.encode.embedding.register_backward_hook(freeze_embedding)
  if FLAGS.num_finetune_chars and FLAGS.use_char and FLAGS.use_char_emb:
    model.encode.char_embedding.register_backward_hook(freeze_char_embedding)

  logging.info('model\n', model)

  param_groups = None
  if FLAGS.lm_path:
    #both BiLanguageModel or RNet or MReader.. use self.ecode so ok update encode.embedding.weight... encode.char_embedding.weight..
    _, updated_params = lele.load(model, FLAGS.lm_path)
    if FLAGS.lm_lr_factor != 1:
      ignored_params = list(map(id, updated_params))
      base_params = filter(lambda p: id(p) not in ignored_params,
                           model.parameters())
      param_groups = [
              {'params': base_params},
              {'params': updated_params, 'lr': FLAGS.learning_rate * FLAGS.lm_lr_factor}
            ]

  train = melt.apps.get_train()

  optimizer = None
  if FLAGS.optimizer == 'noam':
    # Hard to find optim parmas so not to use currently 
    # optimizer = lele.training.optimizers.NoamOpt(FLAGS.emb_dim, 2, 4000,
    #         torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    optimizer = lele.training.optimizers.NoamOpt(128, 2, 4000,
        torch.optim.Adamax(model.parameters(), lr=0))
  
  train(Dataset,
        model,  
        criterion,
        eval_fn=ev.evaluate, 
        valid_write_fn=ev.valid_write,
        infer_write_fn=ev.infer_write,
        valid_names=ev.valid_names,
        valid_suffix='.valid.csv',
        infer_debug_names=ev.valid_names,
        infer_suffix='.infer.txt',
        write_streaming=True,
        optimizer=optimizer,
        param_groups=param_groups
        )   

if __name__ == '__main__':
  tf.app.run()  
