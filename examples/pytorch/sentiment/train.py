#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2018-09-27 19:37:53.455830
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

import pathlib
import random
from collections import Counter, OrderedDict
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm, tnrange

from model import *

nlp = spacy.blank('en')

data_root = pathlib.Path('./data')
df = pd.read_csv(data_root/'Sentiment Analysis Dataset.csv', error_bad_lines=False)

df['SentimentText'] = df.SentimentText.apply(lambda x: x.strip())

words = Counter()
for sent in tqdm(df.SentimentText.values, ascii=True):
    words.update(w.text.lower() for w in nlp(sent))

words = sorted(words, key=words.get, reverse=True)
words = ['_PAD','_UNK'] + words

word2idx = {o:i for i,o in enumerate(words)}
idx2word = {i:o for i,o in enumerate(words)}

def indexer(s): 
  return [word2idx[w.text.lower()] for w in nlp(s)]

df['sentimentidx'] = df.SentimentText.apply(indexer)
df['lengths'] = df.sentimentidx.apply(len)


class VectorizeData(Dataset):
    def __init__(self, df_path, maxlen=10):
        self.maxlen = maxlen
        self.df = pd.read_csv(df_path, error_bad_lines=False)
        self.df['SentimentText'] = self.df.SentimentText.apply(lambda x: x.strip())
        print('Indexing...')
        self.df['sentimentidx'] = self.df.SentimentText.apply(indexer)
        print('Calculating lengths')
        self.df['lengths'] = self.df.sentimentidx.apply(lambda x: self.maxlen if len(x) > self.maxlen else len(x))
        print('Padding')
        self.df['sentimentpadded'] = self.df.sentimentidx.apply(self.pad_data)
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        X = self.df.sentimentpadded[idx]
        lens = self.df.lengths[idx]
        y = self.df.Sentiment[idx]
        return X,y,lens
    
    def pad_data(self, s):
        padded = np.zeros((self.maxlen,), dtype=np.int64)
        if len(s) > self.maxlen: padded[:] = s[:self.maxlen]
        else: padded[:len(s)] = s
        return padded


ds = VectorizeData(data_root/'Sentiment Analysis Dataset.csv')

vocab_size = len(words)
embedding_dim = 4
n_hidden = 5
n_out = 2

m = SimpleGRU(vocab_size, embedding_dim, n_hidden, n_out)

def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X.transpose(0,1), y, lengths # transpose (batch x seq) to (seq x batch)

def fit(model, train_dl, val_dl, loss_fn, opt, epochs=3):
    num_batch = len(train_dl)
    for epoch in tnrange(epochs):      
        y_true_train = list()
        y_pred_train = list()
        total_loss_train = 0
        
        if val_dl:
            y_true_val = list()
            y_pred_val = list()
            total_loss_val = 0
        
        t = tqdm(iter(train_dl), leave=False, total=num_batch, ascii=True)
        for X,y, lengths in t:
            t.set_description(f'Epoch {epoch}')
            X,y,lengths = sort_batch(X,y,lengths)
            X = Variable(X.cuda())
            y = Variable(y.cuda())
            lengths = lengths.numpy()
            
            opt.zero_grad()
            pred = model(X, lengths, gpu=True)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            
            t.set_postfix(loss=loss.data[0])
            pred_idx = torch.max(pred, dim=1)[1]
            
            y_true_train += list(y.cpu().data.numpy())
            y_pred_train += list(pred_idx.cpu().data.numpy())
            total_loss_train += loss.data[0]
            
        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_loss = total_loss_train/len(train_dl)
        print(f' Epoch {epoch}: Train loss: {train_loss} acc: {train_acc}')
        
        if val_dl:
            for X,y,lengths in tqdm(valdl, leave=False, ascii=True):
                X, y,lengths = sort_batch(X,y,lengths)
                X = Variable(X.cuda())
                y = Variable(y.cuda())
                pred = model(X, lengths.numpy())
                loss = loss_fn(pred, y)
                pred_idx = torch.max(pred, 1)[1]
                y_true_val += list(y.cpu().data.numpy())
                y_pred_val += list(pred_idx.cpu().data.numpy())
                total_loss_val += loss.data[0]
            valacc = accuracy_score(y_true_val, y_pred_val)
            valloss = total_loss_val/len(valdl)
            print(f'Val loss: {valloss} acc: {valacc}')


train_dl = DataLoader(ds, batch_size=512)
m = SimpleGRU(vocab_size, embedding_dim, n_hidden, n_out).cuda()
opt = optim.Adam(m.parameters(), 1e-2)

fit(model=m, train_dl=train_dl, val_dl=None, loss_fn=F.nll_loss, opt=opt, epochs=4)

def main(_):
  pass

if __name__ == '__main__':
  tf.app.run()  
  
