#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2018-09-27 19:37:57.468263
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

#flags.DEFINE_string('input', '', '')
#flags.DEFINE_string('output', '', '')

import sys 
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

class SimpleGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_hidden, n_out):
        super().__init__()
        self.vocab_size,self.embedding_dim,self.n_hidden,self.n_out = vocab_size, embedding_dim, n_hidden, n_out
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.n_hidden)
        self.out = nn.Linear(self.n_hidden, self.n_out)
        
    def forward(self, seq, lengths, gpu=True):
        print('Sequence shape',seq.shape)
        print('Lengths',lengths)
        bs = seq.size(1) # batch size
        print('batch size', bs)
        self.h = self.init_hidden(bs, gpu) # initialize hidden state of GRU
        print('Inititial hidden state shape', self.h.shape)
        embs = self.emb(seq)
        embs = pack_padded_sequence(embs, lengths) # unpad
        gru_out, self.h = self.gru(embs, self.h) # gru returns hidden state of all timesteps as well as hidden state at last timestep
        gru_out, lengths = pad_packed_sequence(gru_out) # pad the sequence to the max length in the batch
        print('GRU output(all timesteps)', gru_out.shape)
        print(gru_out)
        print('GRU last timestep output')
        print(gru_out[-1])
        print('Last hidden state', self.h)
        # since it is as classification problem, we will grab the last hidden state
        outp = self.out(self.h[-1]) # self.h[-1] contains hidden state of last timestep
        return F.log_softmax(outp, dim=-1)
    
    def init_hidden(self, batch_size, gpu):
        if gpu: return Variable(torch.zeros((1,batch_size,self.n_hidden)).cuda())
        else: return Variable(torch.zeros((1,batch_size,self.n_hidden)))
