#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   torch-train.py
#        \author   chenghuige  
#          \date   2019-08-02 01:05:59.741965
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

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pyt.dataset import *
from dataset import *
from pyt.model import *
import pyt.model as base
import evaluate as ev
import loss

import melt
logging = melt.logging
import gezi
import lele

#import horvod.torch as hvd
# hvd.init()
# # Horovod: pin GPU to local rank.
# torch.cuda.set_device(hvd.local_rank())
# seed = 1024
# torch.cuda.manual_seed(seed)


def main(_):
  FLAGS.torch_only = True
  melt.init()
  fit = melt.get_fit()

  FLAGS.eval_batch_size = 512 * FLAGS.valid_multiplier

  model_name = FLAGS.model
  model = getattr(base, model_name)() 

  loss_fn = nn.BCEWithLogitsLoss()

  td = TextDataset()
  train_files = gezi.list_files('../input/train/*')
  train_ds = get_dataset(train_files, td)
  
  import multiprocessing
  #--easy to be Killed .. if large workers
  num_threads = int(multiprocessing.cpu_count() * 0.3)
  logging.info('num_threads as multiprocessing.cpu_count', num_threads)
 
  num_threads = 12 
  train_dl = DataLoader(train_ds, FLAGS.batch_size, shuffle=True, num_workers=num_threads, collate_fn=lele.DictPadCollate())
  #logging.info('num train examples', len(train_ds), len(train_dl))
  valid_files = gezi.list_files('../input/valid/*')
  valid_ds = get_dataset(valid_files, td)
  valid_dl = DataLoader(valid_ds, FLAGS.eval_batch_size, collate_fn=lele.DictPadCollate(), num_workers=num_threads)
  valid_dl2 = DataLoader(valid_ds, FLAGS.batch_size, collate_fn=lele.DictPadCollate(), num_workers=num_threads)
  #logging.info('num valid examples', len(valid_ds), len(valid_dl))

  # fit(model,  
  #     loss_fn,
  #     dataset=train_dl,
  #     valid_dataset=valid_dl,
  #     valid_dataset2=valid_dl2,
  #     eval_fn=ev.evaluate,
  #     valid_write_fn=ev.valid_write,
  #     #write_valid=FLAGS.write_valid)   
  #     write_valid=False,
  #    )


if __name__ == '__main__':
  tf.app.run()  
  
