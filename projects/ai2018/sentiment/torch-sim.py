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
  
#tf.enable_eager_execution()

import numpy as np
from tqdm import tqdm

import melt 
logging = melt.logging
import gezi
import traceback

from wenzheng.utils import input_flags 

import torch_algos.model as base
from dataset import Dataset
import lele

from prepare.filter import filter
from wenzheng.utils import ids2text
import numpy as np
import jieba

from algos.config import ATTRIBUTES
import evaluate as ev

vocab = None
char_vocab = None
model = None

def convert(content):
  content = filter(content)
  words = [word for word in jieba.cut(content)]
  words = gezi.add_start_end(words, '<S>', '</S>')

  wids =  [vocab.id(word) for word in words]
  chars = [list(word) for word in words]
  char_ids = np.zeros([len(wids), FLAGS.char_limit], dtype=np.int32)
  for i, token in enumerate(chars):
    for j, ch in enumerate(token):
      if j == FLAGS.char_limit:
        break
      char_ids[i, j] = char_vocab.id(ch)

  char_ids = char_ids.reshape(-1)

  input = {'content': np.array([wids], dtype=np.long), 'char': np.array([char_ids], dtype=np.long)}
  return lele.to_torch(input), words

def predict(content):
  x, words = convert(content)
  logits = model(x)[0]
  logits = logits.detach().cpu().numpy()
  probs = gezi.softmax(logits, 1)
  #print(probs)
  print(list(zip(ATTRIBUTES, [list(x) for x in probs])))

  predicts = np.argmax(logits, -1) - 2
  print('predicts ', predicts)
  print(list(zip(ATTRIBUTES, predicts)))
  # adjusted_predicts = ev.to_predict(logits)
  # print('apredicts', adjusted_predicts)
  # print(list(zip(ATTRIBUTES, adjusted_predicts)))

  alpha = model.pooling.poolings[0].alpha
  alpha = alpha.detach().cpu().numpy()
  for i, attr in enumerate(ATTRIBUTES):
    scores = alpha[0, i]
    print(attr, predicts[i])
    for word, score in zip(words, scores):
      print(word, score * len(words))

def encode(content):
  x, _ = convert(content)
  _ = model(x)
  feature = model.feature[0]
  feature = feature.detach().cpu().numpy()
  return feature

def sim(content1, content2):
  f1 = encode(content1)
  f2 = encode(content2)
  score = gezi.cosine(f1, f2)
  print(content1, content2, score)

def main(_):
  FLAGS.torch = True
  FLAGS.emb_dim = 300 
  FLAGS.cell = 'lstm'
  FLAGS.num_layers = 1
  FLAGS.rnn_hidden_size = 400
  FLAGS.concat_layers = False
  FLAGS.use_char = True

  FLAGS.share_fc = True
  FLAGS.share_pooling = True
  FLAGS.recurrent_dropout = False 
  FLAGS.rnn_no_padding = False 
  FLAGS.rnn_padding = True 
  FLAGS.att_combiner = 'sfu'
  FLAGS.hop = 1
  FLAGS.use_label_att = False 
  FLAGS.use_self_match = False 
  FLAGS.encoder_type = 'rnn'
  FLAGS.encoder_output_method = 'max'

  FLAGS.model_dir = '/home/gezi/temp/ai2018/sentiment/model/lm/word.jieba.ft.long/torch.word.lm.nopad.lstm.hidden400/'

  melt.apps.init()

  FLAGS.model = 'MReader'
  model_path = '%s/latest.pyt' % FLAGS.model_dir
  global vocab, char_vocab, model
  vocab_path = '%s/vocab.txt' % FLAGS.model_dir
  char_vocab_path = vocab_path.replace('vocab.txt', 'char_vocab.txt')
  FLAGS.vocab = vocab_path
  vocab = gezi.Vocabulary(vocab_path)
  char_vocab = gezi.Vocabulary(char_vocab_path)
  
  model = getattr(base, FLAGS.model)()
  model = model.cuda()

  print('model\n', model)
  lele.load(model, model_path)

  contents = ['这是一个很好的餐馆，菜很不好吃，我还想再去', 
              '这是一个很差的餐馆，菜很不好吃，我不想再去', 
              '这是一个很好的餐馆，菜很好吃，我还想再去', 
              '这是一个很好的餐馆，只是菜很难吃，我还想再去',
              '这是一个很好的餐馆，只是菜很不好吃，我还想再去',
              '好吃的！黑胡椒牛肉粒真的是挺咸的',
              '不论是环境的宽敞度还是菜的味道上',
              '烤鸭皮酥脆入口即化，总体还可以',
              '烤鸭皮酥脆入口即化',
              '软炸鲜菇据说是他家的优秀美味担当',
              '环境挺好的，服务很到位',
              '肉松的味道都不错，量一般',
              '也不算便宜，不过吃着好吃',
              '高大上的餐厅，一级棒的环境',
              '比较硬，比较喜欢芝士和咖喱口味的',
              '有嚼劲，很入味宫廷豌豆黄造型可爱',
              '蔬菜中规中矩，解腻不错',
              '女友生日菜有点贵架势不错味道就这样',
              '相比其他兰州拉面粗旷的装饰风格，这家设计很小清新，座位宽敞，客人不多']
  
  sim('牛肉赞', '牛肉好吃')
  sim('牛肉赞', '牛肉不好吃')
  sim('牛肉赞', '牛肉一般')
  sim('牛肉赞', '他们家的牛肉很好吃')
  sim('牛肉赞', '羊肉一般')
  sim('牛肉赞', '羊肉好吃')
  sim('羊肉赞', '羊肉好吃')

  sim('适合闺蜜聚会', '适合朋友聚会')
  sim('适合闺蜜聚会', '适合孕妇的餐厅')
  sim('适合闺蜜聚会', '适合拍照的餐厅')
  sim('适合闺蜜聚会', '和闺蜜一起来的')

  sim('当来到颐和园，皇家所独有的庄重大气便被融入内核', '在我印象应该不仅是豪华酒店了，而是奢华酒店级别的了')
  sim('店非常的简约小清新', '在我印象应该不仅是豪华酒店了，而是奢华酒店级别的了')
  sim('店非常的简约小清新', '环境小清新')
  sim('店非常的简约小清新', '文艺清新')
  sim('环境小清新', '文艺清新')
  sim('环境小清新', '高大上')
  sim('当来到颐和园，皇家所独有的庄重大气便被融入内核', '高大上')
  sim('当来到颐和园，皇家所独有的庄重大气便被融入内核', '文艺清新')
  sim('在我印象应该不仅是豪华酒店了，而是奢华酒店级别的了', '高大上')
  sim('在我印象应该不仅是豪华酒店了，而是奢华酒店级别的了', '文艺清新')
  sim('在我印象应该不仅是豪华酒店了，而是奢华酒店级别的了', '适合拍照')
  sim('在我印象应该不仅是豪华酒店了，而是奢华酒店级别的了', '复古')
  sim('古色古香的院子', '复古')

  # # print words importance scores
  # word_scores_list = model.pooling.word_scores

  # for word_scores in word_scores_list:
  #   print(list(zip(words, word_scores[0].numpy())))


if __name__ == '__main__':
  tf.app.run()  
