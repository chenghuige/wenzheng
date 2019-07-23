#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   infer.py
#        \author   chenghuige  
#          \date   2019-07-23 09:24:50.235252
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

from absl import app as absl_app
from absl import flags
FLAGS = flags.FLAGS

import pandas as pd
from tqdm import tqdm
import cv2
from keras.models import load_model
import numpy as np
from sklearn.metrics import f1_score, fbeta_score, cohen_kappa_score, confusion_matrix

# TODO can load all info like SIZE from saved model meta data ?
SIZE = 300
NUM_CLASSES= 5

class Predictor():
  def __init__(self, model, batch_size, predict_fn=None):
    self.predicted = []
    self.inputs = []
    self.model = model
    self.batch_size = batch_size
    self.predict_fn = predict_fn

  def _predict(self):
    self.inputs = np.array(self.inputs)
    if self.predict_fn:
      self.predicted += self.predict_fn(self.model, self.inputs)
    else:
      self.predicted += list(self.model.predict(self.inputs))
    self.inputs = []

  def add(self, x):
    self.inputs.append(x)
    if len(self.inputs) == self.batch_size:
      self._predict()
    
  def predict(self):
    if self.inputs:
      self._predict()
    return np.array(self.predicted)

def hack_lb(test_preds):
  id_codes = np.load('../input/aptos2019/aptos2019-test/small_id_codes.npy', allow_pickle = True)
  small_ids_df = pd.DataFrame(id_codes, columns=["id_code"])

  test_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
  sample_df = small_ids_df
  sample_df["diagnosis"] = test_preds
  sub = pd.merge(test_df, sample_df, on='id_code', how='left').fillna(0)
  sub["diagnosis"] = sub["diagnosis"].astype(int)
  return sub
      
def main(_): 
  if not FLAGS.valid:
    if not FLAGS.lb_only:
      submit = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
    else:
      submit = pd.read_csv('../input/aptos2019/aptos2019-test/test.csv')
  else:
    submit = pd.read_csv('../input/aptos2019-blindness-detection/train_%d.csv' % FLAGS.fold)
  
  model_path = '../input/aptos2019/densenet_bestqwk.h5' if not FLAGS.model_path else FLAGS.model_path
  print('loading model from ', model_path)
  try:
    model = load_model(model_path)
  except Exception:
    # https://github.com/keras-team/keras/issues/5916
    import loss
    model = load_model(model_path, custom_objects={'kappa_loss': loss.kappa_loss})
 
  predict_fn = None
  if FLAGS.tta:
    def predict_fn_(model, X):
      # now only cosinder classification TODO
      score =((model.predict(X).ravel()*model.predict(X[:, ::-1, :, :]).ravel()*model.predict(X[:, ::-1, ::-1, :]).ravel()*model.predict(X[:, :, ::-1, :]).ravel())**0.25).reshape([-1, NUM_CLASSES]).tolist()
      return score
    predict_fn = predict_fn_ 

  print('predict_fn', predict_fn)
  predictor = Predictor(model, FLAGS.batch_size, predict_fn)

  for name in tqdm(submit['id_code'].values, ascii=True):
    if not FLAGS.valid:
      path = os.path.join('../input/aptos2019-blindness-detection/test_images/', name + '.png')
    else:
      path = os.path.join('../input/aptos2019-blindness-detection/train_images/', name + '.png')

    image = cv2.imread(path)
    image = cv2.resize(image, (SIZE, SIZE))

    image = image / 255
    
    predictor.add(image)

  predicted = predictor.predict()
  
  ## classification
  predicted = np.array([np.argmax(x) for x in predicted])

  ## ordinal classification
  #predicted = np.array([np.sum(x > 0.5, axis=-1)  for x in predicted])
  
  ## regression
  #predicted = np.array([int(x + 0.5)  for x in predicted])

  if not FLAGS.valid:
    if not FLAGS.lb_only:
      submit['diagnosis'] = predicted
    else:
      submit = hack_lb(predicted)
  else:
    submit['predict'] = predicted
    score = cohen_kappa_score(submit['diagnosis'].values,
                              submit['predict'].values,
                              labels=[0,1,2,3,4],
                              weights='quadratic')
    print('kappa score for valid file', score)

  result_file = 'submission.csv'
  if FLAGS.model_path:
    dir = os.path.dirname(FLAGS.model_path)
    result_file = '{}/{}'.format(dir, result_file)

  submit.to_csv(result_file, index=False)
  submit.head()


if __name__ == '__main__':
  flags.DEFINE_integer('batch_size', 128, '')
  flags.DEFINE_string('model_path', None, '')
  flags.DEFINE_bool('tta', False, '')
  flags.DEFINE_bool('optimize_kappa', False, '')
  flags.DEFINE_bool('valid', False, 'infer valid or infer test')
  flags.DEFINE_integer('fold', 0, '')
  flags.DEFINE_bool('lb_only', False, 'only consider lb score')

  absl_app.run(main) 
