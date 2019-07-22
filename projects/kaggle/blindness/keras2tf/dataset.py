#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2019-07-21 21:52:09.914646
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from tensorflow.keras.utils import Sequence
import numpy as np
from aug import seq 
from sklearn.utils import shuffle
import cv2

from config import *

class Dataset(Sequence):
  
  def __init__(self, image_filenames, labels,
                batch_size, is_train=True,
                mix=False, augment=False):
    self.image_filenames, self.labels = image_filenames, labels
    self.batch_size = batch_size
    self.is_train = is_train
    self.is_augment = augment
    if(self.is_train):
        self.on_epoch_end()
    self.is_mix = mix

  def __len__(self):
    return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

  def __getitem__(self, idx):
    batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

    if(self.is_train):
        return self.train_generate(batch_x, batch_y)
    return self.valid_generate(batch_x, batch_y)

  def on_epoch_end(self):
    if(self.is_train):
        self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)
    else:
        pass
  
  def mix_up(self, x, y):
    lam = np.random.beta(0.2, 0.4)
    ori_index = np.arange(int(len(x)))
    index_array = np.arange(int(len(x)))
    np.random.shuffle(index_array)        
    
    mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]
    mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]
    
    return mixed_x, mixed_y

  def train_generate(self, batch_x, batch_y):
    batch_images = []
    for (sample, label) in zip(batch_x, batch_y):
        img = cv2.imread('../input/aptos2019-blindness-detection/train_images/'+sample+'.png')
        img = cv2.resize(img, (SIZE, SIZE))
        if(self.is_augment):
            img = seq.augment_image(img)
        batch_images.append(img)
    batch_images = np.array(batch_images, np.float32) / 255
    batch_y = np.array(batch_y, np.float32)
    if(self.is_mix):
        batch_images, batch_y = self.mix_up(batch_images, batch_y)
    return batch_images, batch_y

  def valid_generate(self, batch_x, batch_y):
    batch_images = []
    for (sample, label) in zip(batch_x, batch_y):
        img = cv2.imread('../input/aptos2019-blindness-detection/train_images/'+sample+'.png')
        img = cv2.resize(img, (SIZE, SIZE))
        batch_images.append(img)
    batch_images = np.array(batch_images, np.float32) / 255
    batch_y = np.array(batch_y, np.float32)
    return batch_images, batch_y  
