#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   optimizers.py
#        \author   chenghuige  
#          \date   2018-10-29 07:06:55.090940
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

import sys 
import os
import melt

# http://nlp.seas.harvard.edu/2018/04/03/attention.html

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self.param_groups = self.optimizer.param_groups
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
      self.optimizer.zero_grad()

    def state_dict(self):
      return self.optimizer.state_dict()

    def load_state_dict(self, x):
        return self.optimizer.load_state_dict(x)
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))  

def lr_poly(base_lr, iter, max_iter, end_learning_rate, power):
    return max((base_lr - end_learning_rate) * ((1-float(iter)/max_iter)**(power)), end_learning_rate)

class BertOpt:
    "Optim wrapper that implements rate."
    def __init__(self, lr, min_lr, num_train_steps, warmup, optimizer):
        self.optimizer = optimizer
        import melt
        self._step = 0
        self.warmup = warmup
        self._rate = 0
        self.lr = lr
        self.ori_min_lr = min_lr
        self.min_lr = min_lr
        self.num_train_steps = num_train_steps
        self.param_groups = self.optimizer.param_groups
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
            
        warmup_percent_done = step / self.warmup
        warmup_learning_rate = self.lr * warmup_percent_done

        # decay by eval value ?
       
        if melt.epoch() >= 9 and FLAGS.num_epochs > 9:
          self.min_lr = self.ori_min_lr * ((FLAGS.num_epochs - melt.epoch()) / (FLAGS.num_epochs - 5))
        #print('-----------------', melt.epoch(), melt.epoch() > 9, self.min_lr, self.ori_min_lr)
        
        is_warmup = step < self.warmup
        learning_rate = lr_poly(self.lr, step, self.num_train_steps, self.min_lr, 1.)
        learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
        #print('-----------------', is_warmup, warmup_percent_done, warmup_learning_rate, warmup_learning_rate)
        return learning_rate

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, x):
        return self.optimizer.load_state_dict(x)

if __name__ == '__main__':
  import melt
  import matplotlib.pyplot as plt 
  import numpy as np

  opts = [NoamOpt(512, 1, 4000, None), 
          NoamOpt(512, 1, 8000, None),
          NoamOpt(256, 1, 4000, None),
          NoamOpt(200, 2, 4000, None),
          NoamOpt(256, 2, 4000, None),
          NoamOpt(300, 2, 4000, None),
          NoamOpt(128, 2, 4000, None)]
  plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
  plt.legend(["512:4000", "512:8000", "256:4000", "200:2:4000",  "256:2:4000", "300:2:4000", "128:2:4000"])

  for i in range(1, 40000, 1000):
      print(i, NoamOpt(200, 2, 2000, None).rate(i))

  plt.savefig('/home/gezi/tmp/lr.png')           
