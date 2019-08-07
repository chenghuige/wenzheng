#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   evaluate.py
#        \author   chenghuige  
#          \date   2019-07-21 21:52:01.775723
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf
from keras.callbacks import Callback

import numpy as np
from sklearn.metrics import f1_score, fbeta_score, cohen_kappa_score, confusion_matrix

from config import *

import cv2
from gezi import SummaryWriter
import gezi

import seaborn as sns
from io import BytesIO  
import matplotlib.pyplot as plt

from dataset import image_dict

def gen_confusion(y_true, y_pred, info=''):
  cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  fig, ax = plt.subplots()
  sns.heatmap(cm, annot=True)
  ax.set(ylabel='True label', xlabel='Predicted label,{}'.format(info))
  s = BytesIO()
  plt.savefig(s, format='png', bbox_inches='tight')
  return s

def to_str(scores):
  return ','.join(['{:.2f}'.format(x) for x in scores])

class Evaluator(Callback):
  def __init__(self, 
                dir,
                validation_data=(), 
                interval=1, 
                loss_type='classification'):
    super(Callback, self).__init__()

    self.dir = dir
    self.interval = interval
    self.valid_generator, self.y_val = validation_data
    self.history = []
    self.epoch = 0

    self.loss_type = loss_type
  
    self.logger = SummaryWriter(self.dir)
    
  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.interval == 0:
      y_pred = self.model.predict_generator(generator=self.valid_generator,
                                            workers=1, 
                                            use_multiprocessing=False,
                                            verbose=1)

      
      #print(y_pred)
      #print('----------', self.y_val.shape, y_pred.shape)
      
      y_scores = y_pred
      
      if 'classification' in self.loss_type:
        if not 'ordinal' in self.loss_type:
          def flatten(y):
            return np.argmax(y, axis=1).reshape(-1)
        else:
          if not '2' in self.loss_type:
            def flatten(y):
              return np.sum(y > 0.5, axis=1).reshape(-1) - 1
          else:
            def flatten(y):
              return np.sum(y > 0.5, axis=1).reshape(-1) 

        y_true = flatten(self.y_val)
        y_pred = flatten(y_pred)
      else: #regression
        y_true = self.y_val.reshape(-1)
        y_pred = y_pred.reshape(-1)

        if not 'sigmoid2' in self.loss_type:
          def trans(x):
            return int(x + 0.5)
            # if x >= NUM_CLASSES:
            #   return NUM_CLASSES - 1
            # if x <= 0:
            #   return 0
            # return int(x)
        else:
          
          def trans(x):
            return int(x * 4.0 + 0.5)

          def trans2(x):
            return int(x * 4.0)
          
          y_true = np.array([trans2(x) for x in y_true])

        y_pred = np.array([trans(x) for x in y_pred])
        
      if 'classification' in self.loss_type:
        y_score = []
        for scores, pred in zip(y_scores, y_pred):
          if not 'ordinal2' in self.loss_type:
            y_score.append(scores[pred])
          else:
            y_score.append(scores[max(pred - 1, 0)])
        y_score = np.array(y_score)
      else:
        y_score = y_scores.reshape(-1)


      score = cohen_kappa_score(y_true,
                                y_pred,
                                labels=[0,1,2,3,4],
                                weights='quadratic')
      print("\n epoch: %d - kappa: %.6f \n" % (epoch + 1, score))

      self.logger.scalar('kappa', score, epoch + 1)

      print('image_dict size', len(image_dict))

      batch_idx = 0
      batch_images, _ = self.valid_generator[batch_idx]
      idx = batch_idx * self.valid_generator.batch_size
      texts = []
      timer = gezi.Timer('image/rand')
      indexes = [x + idx for x in range(20)]
      for i in range(len(indexes)):
        texts.append('id:{}\nlabel:{}\npred:{}\nscore:{:.3f}\nscores:{}'.format(
                      self.valid_generator.image_filenames.values[indexes[i]], 
                      y_true[indexes[i]], 
                      y_pred[indexes[i]],
                      y_score[indexes[i]],
                      to_str(y_scores[indexes[i]]))) 
      images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.valid_generator.get_images(indexes)]       
      
      self.logger.image('rand', images, epoch + 1, texts)
      timer.print()

      # timer = gezi.Timer('image/confidence')
      # #print(y_score.shape, y_true.shape, y_pred.shape)
      # indexes = (- y_score * (y_true != y_pred)).argsort()[:20]
      # texts = []
      # #print('---------------', indexes)
      # for i in range(len(indexes)):
      #   texts.append('id:{}\nlabel:{}\npred:{}\nscore:{:.3f}\nscores:{}'.format(
      #                 self.valid_generator.image_filenames.values[indexes[i]], 
      #                 y_true[indexes[i]], 
      #                 y_pred[indexes[i]],
      #                 y_score[indexes[i]],
      #                 to_str(y_scores[indexes[i]]))) 
      # images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.valid_generator.get_images(indexes)]  

      # self.logger.image('confidence', images, epoch + 1, texts)  
      # timer.print()

      timer = gezi.Timer('image/dist')
      indexes = (-abs(y_true - y_pred)).argsort()[:20]
      texts = []
      for i in range(len(indexes)):
        texts.append('id:{}\nlabel:{}\npred:{}\nscore:{:.3f}\nscores:{}'.format(
                      self.valid_generator.image_filenames.values[indexes[i]], 
                      y_true[indexes[i]], 
                      y_pred[indexes[i]],
                      y_score[indexes[i]],
                      to_str(y_scores[indexes[i]]))) 
      images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.valid_generator.get_images(indexes)]   

      self.logger.image('dist', images, epoch + 1, texts)   
      timer.print()

      confusion = gen_confusion(y_true, y_pred, info='kappa:{:.4f}'.format(score))
      self.logger.image('confusion', confusion, epoch + 1, bytes_input=True)

      self.history.append(score)
      if score >= max(self.history):
        timer = gezi.Timer('saving checkpoint with current best kappa:{:.4f}'.format(score))
        self.model.save('{}/densenet_bestqwk.h5'.format(self.dir))
        timer.print()

