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

import torch.optim as optim
import torch.utils.data.distributed
import horovod.torch as hvd

import numpy as np

from pyt.dataset import *
import text_dataset
from pyt.model import *
import pyt.model as base
import evaluate as ev
import loss

import melt
logging = melt.logging
import gezi
import lele

logging.init('/home/gezi/temp')

import horovod.torch as hvd
hvd.init()
# Horovod: pin GPU to local rank.
torch.cuda.set_device(hvd.local_rank())
seed = 1024
torch.cuda.manual_seed(seed)

def train(epoch, model, loss_fn, train_loader, optimizer):
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    train_loader.sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        for key in data:
          if type(data[key][0]) != np.str_:
            data[key] = data[key].cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def test(model, loss_fn, test_loader):
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += loss_fn(output, target)

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler)

    # Horovod: average metric values across workers.
    test_loss = metric_average(test_loss, 'avg_loss')

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


def main(_):
  FLAGS.torch_only = True
  melt.init()
  #fit = melt.get_fit()

  FLAGS.eval_batch_size = 512 * FLAGS.valid_multiplier

  model_name = FLAGS.model
  model = getattr(base, model_name)() 

  model = model.cuda()

  loss_fn = nn.BCEWithLogitsLoss()

  td = text_dataset.Dataset()

  train_files = gezi.list_files('../input/train/*')
  train_ds = get_dataset(train_files, td)
  
  kwargs = {'num_workers': 4, 'pin_memory': True, 'collate_fn': lele.DictPadCollate()}
  #kwargs = {'num_workers': 1, 'pin_memory': False, 'collate_fn': lele.DictPadCollate()}

  train_sampler = train_ds
  train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_ds, num_replicas=hvd.size(), rank=hvd.rank())
  
  train_dl = DataLoader(train_ds, FLAGS.batch_size, sampler=train_sampler, **kwargs)

  logging.info('train dl done')
  
  valid_files = gezi.list_files('../input/valid/*')
  valid_ds = get_dataset(valid_files, td)

  # support shuffle=False from version 1.2
  valid_sampler = torch.utils.data.distributed.DistributedSampler(
    valid_ds, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False)
  logging.info('valid sampler done')
  valid_sampler2 = torch.utils.data.distributed.DistributedSampler(
    valid_ds, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False)
  logging.info('valid sampler2 done')
  
  valid_dl = DataLoader(valid_ds, FLAGS.eval_batch_size, sampler=valid_sampler, **kwargs)
  logging.info('valid dl done')
  valid_dl2 = DataLoader(valid_ds, FLAGS.batch_size, sampler=valid_sampler2, **kwargs)
  logging.info('valid dl2 done')


  optimizer = optim.SGD(model.parameters(), lr=0.1)
  hvd.broadcast_parameters(model.state_dict(), root_rank=0)
  hvd.broadcast_optimizer_state(optimizer, root_rank=0)

  logging.info('optimizer broadcast done')

  optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters())

   logging.info('optimizer hvd done, now train')
  for epoch in range(2):
    train(epoch, model, loss_fn, train_dl, optimizer)
    test(model, loss_fn, valid_dl)

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
  
