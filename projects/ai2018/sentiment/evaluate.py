#!/usr/bin/env python
# ==============================================================================
#          \file   read-records.py
#        \author   chenghuige  
#          \date   2016-07-19 17:09:07.466651
#   \Description  
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, time
import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('info_path', None, '')
flags.DEFINE_bool('auc_need_softmax', True, '')

flags.DEFINE_string('class_weights_path', './mount/temp/ai2018/sentiment/class_weights.npy', '')
flags.DEFINE_float('logits_factor', 10, '10 7239 9 7245 but test set 72589 and 72532 so.. a bit dangerous')

flags.DEFINE_bool('show_detail', False, '')

flags.DEFINE_string('i', '.', '')

flags.DEFINE_string('metric_name', 'adjusted_f1/mean', '')
flags.DEFINE_float('min_thre', 0., '0.705')
flags.DEFINE_integer('len_thre', 256, '')
flags.DEFINE_float('max_thre', 1000., '')
flags.DEFINE_bool('adjust', True, '')
flags.DEFINE_bool('more_adjust', True, '')

#from sklearn.utils.extmath import softmax
from sklearn.metrics import f1_score, log_loss, roc_auc_score

from melt.utils.weight_decay import WeightDecay, WeightsDecay

import numpy as np
import glob
import gezi
import melt 
logging = melt.logging

from wenzheng.utils import ids2text
#from projects.ai2018.sentiment.algos.config import ATTRIBUTES, NUM_ATTRIBUTES, NUM_CLASSES, CLASSES
from algos.config import ATTRIBUTES, NUM_ATTRIBUTES, NUM_CLASSES, CLASSES


import pickle
import pandas as pd
import traceback

#since we have same ids... must use valid and test 2 infos
valid_infos = {}
test_infos = {}

decay = None
wnames = []
classes = ['0na', '1neg', '2neu', '3pos']
num_classes = len(classes)

class_weights = None

def load_class_weights():
  global class_weights
  if class_weights is None:
    if FLAGS.adjust:
      if not os.path.exists(FLAGS.class_weights_path):
        FLAGS.class_weights_path = '/home/gezi/temp/ai2018/sentiment/class_weights.npy'
        if not os.path.exists(FLAGS.class_weights_path):
          FLAGS.class_weights_path = './class_weights.npy'
      class_weights = np.load(FLAGS.class_weights_path)
      for i in range(len(class_weights)):
        for j in range(num_classes):
          x = class_weights[i][j]
          class_weights[i][j] = x * x * x 
      
      if FLAGS.more_adjust:
        class_weights[1][-2] = class_weights[1][-2] * pow(1.2, 22)
        class_weights[-2][0] = class_weights[-2][0] * 60000
    else:
      class_weights = np.ones([NUM_ATTRIBUTES, NUM_CLASSES])
  return class_weights

def init():
  global valid_infos, test_infos
  global wnames
  load_class_weights()
  
  with open(FLAGS.info_path, 'rb') as f:
    valid_infos = pickle.load(f)
  if FLAGS.test_input:
    with open(FLAGS.info_path.replace('.pkl', '.test.pkl'), 'rb') as f:
      test_infos = pickle.load(f)

  ids2text.init()

  #min_learning_rate = 1e-5
  min_learning_rate = FLAGS.min_learning_rate
  if FLAGS.decay_target:
    global decay
    decay_target = FLAGS.decay_target
    cmp = 'less' if decay_target == 'loss' else 'greater'
    if FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES * NUM_CLASSES:
      for attr in ATTRIBUTES:
        for j, cs in enumerate(CLASSES):
          wnames.append(f'{attr}_{j}{cs}')
    elif FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES:
      wnames = ATTRIBUTES
    if not decay:
      logging.info('Weight decay target:', decay_target)
      if FLAGS.num_learning_rate_weights <= 1:
        decay = WeightDecay(patience=FLAGS.decay_patience, 
                      decay=FLAGS.decay_factor, 
                      cmp=cmp,
                      decay_start_epoch=FLAGS.decay_start_epoch_,
                      min_learning_rate=min_learning_rate)
      else:
        decay = WeightsDecay(patience=FLAGS.decay_patience, 
                      decay=FLAGS.decay_factor, 
                      cmp=cmp,
                      min_learning_rate=min_learning_rate,
                      decay_start_epoch=FLAGS.decay_start_epoch_,
                      names=wnames)  


# def to_predict(logits):
#   probs = gezi.softmax(logits, 1)
#   result = np.zeros([len(probs)], dtype=np.int32)
#   for i, prob in enumerate(probs):
#     # it can imporve but depends 0.6 or 0.7  or 0.8 ?
#     if prob[0] >= 0.6:
#       result[i] = -2
#     else:
#       result[i] = np.argmax(prob[1:]) - 1

#   return result

def regression_to_class(predict):
  if predict > 7:
    return 3
  elif predict > 5:
    return 2
  elif predict > 3:
    return 1
  else:
    return 0

def to_class(predicts, thre=0.5):
  if FLAGS.loss_type == 'hier':
    ## TODO even hier.. still not good below...
    # result = np.zeros([len(predicts)], dtype=np.int32)
    # for i, predict in enumerate(predicts):
    #   na_prob = gezi.sigmoid(predict[0])
    #   if na_prob > thre:
    #     result[i] = 0
    #   else:
    #     result[i] = np.argmax(predict[1:]) + 1
    # return result
    return np.argmax(predicts, -1)
  elif FLAGS.loss_type == 'regression':
    return np.array([regression_to_class(x) for x in predicts])
  else:
    return np.argmax(predicts, -1)
  
def calc_f1(labels, predicts, model_path=None, name = 'f1'):
  names = ['mean'] + ATTRIBUTES + classes
  num_classes = NUM_CLASSES if FLAGS.binary_class_index is None else 2
  if FLAGS.binary_class_index is not None:
    names = ['mean'] + ATTRIBUTES + ['not', classes[FLAGS.binary_class_index]] 

  names = [f'{name}/' + x for x in names]
  # TODO show all 20 * 4 ? not only show 20 f1
  f1_list = []
  class_f1 = np.zeros([num_classes])
  all_f1 = []
  for i in range(NUM_ATTRIBUTES):
    #f1 = f1_score(labels[:,i], np.argmax(predicts[:,i], 1) - 2, average='macro')
    # print(labels[:,i])
    # print(predicts[:,i])
    # print(len(labels[:,i]))
    # print(len(predicts[:,i]))

    scores = f1_score(labels[:,i], to_class(predicts[:,i]), average=None)
    
    # if FLAGS.binary_class_index is not None:
    #   scores = [scores[1]]
    ## this will be a bit better imporve 0.001, I will just use when ensemble
    #scores = f1_score(labels[:,i], to_predict(predicts[:,i]), average=None)
    class_f1 += scores
    all_f1 += list(scores)
    f1 = np.mean(scores)
    f1_list.append(f1)
  f1 = np.mean(f1_list)
  class_f1 /= NUM_ATTRIBUTES

  vals = [f1] + f1_list + list(class_f1)

  if model_path is None:
    if FLAGS.decay_target and FLAGS.decay_target == name:
      if  FLAGS.num_learning_rate_weights <= 1:
        target = f1
      elif FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES * NUM_CLASSES:
        target = all_f1
      elif FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES:
        target = f1_list
      else:
        raise f'Unsupported weights number{FLAGS.num_learning_rate_weights}'
 
      weights = decay.add(target)
      if FLAGS.num_learning_rate_weights > 1:
        vals += list(weights)
        names += [f'weights/{name}' for name in wnames]

  return vals, names

def calc_loss(labels, predicts, model_path=None):
  """
  softmax loss, mean loss and per attr loss
  """
  names = ['mean'] + ATTRIBUTES
  names = ['loss/' + x for x in names]
  losses = []
  for i in range(NUM_ATTRIBUTES):
    loss = log_loss(labels[:,i], predicts[:,i])
    losses.append(loss)
  loss = np.mean(losses)
  vals = [loss] + losses

  if model_path is None:
    if FLAGS.decay_target and FLAGS.decay_target == 'loss':
      if  FLAGS.num_learning_rate_weights <= 1:
        target = loss
      elif FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES:
        target = losses
      else:
        raise f'Unsupported weights number{FLAGS.num_learning_rate_weights}'

      weights = decay.add(target)
      if FLAGS.num_learning_rate_weights > 1:
        vals += list(weights)
        names += [f'weights/{name}' for name in wnames]

  return vals, names

# TODO understand macro micro..
def calc_auc(labels, predicts, model_path=None):
  """
  per attr auc
  """
  names = ['mean'] + ATTRIBUTES + classes
  names = ['auc/' + x for x in names]
  aucs_list = []
  class_aucs = np.zeros([num_classes])
  for i in range(NUM_ATTRIBUTES):
    aucs = []
    #print(np.sum(predicts[:,i], -1))
    for j in range(NUM_CLASSES):
      auc = roc_auc_score((labels[:, i] == j).astype(int), predicts[:, i, j])
      aucs.append(auc)
    auc = np.mean(aucs) 
    aucs_list.append(auc)
    class_aucs += np.array(aucs)
  class_aucs /= NUM_ATTRIBUTES
  auc = np.mean(aucs_list)
  vals = [auc] + aucs_list + list(class_aucs)

  if model_path is None:
    if FLAGS.decay_target and FLAGS.decay_target == 'auc':
      if  FLAGS.num_learning_rate_weights <= 1:
        target = auc
      elif FLAGS.num_learning_rate_weights == NUM_ATTRIBUTES:
        target = aucs_list
      else:
        raise f'Unsupported weights number{FLAGS.num_learning_rate_weights}'

      weights = decay.add(target)
      if FLAGS.num_learning_rate_weights > 1:
        vals += list(weights)
        names += [f'weights/{name}' for name in wnames]

  return vals, names

def evaluate(labels, predicts, ids=None, model_path=None):
  # TODO here use softmax will cause problem... not correct.. for f1
  probs = gezi.softmax(predicts)

  #adjusted_probs = gezi.softmax(predicts * FLAGS.logits_factor) * class_weights * [1, 4., 5., 1.]
  adjusted_probs = gezi.softmax(predicts * FLAGS.logits_factor) * class_weights 

  mean_vals = []
  mean_names = []

  #vals, names = calc_f1(labels, predicts, model_path)
  vals, names = calc_f1(labels, probs, model_path)

  mean_vals.append(vals[0])
  mean_names.append(names[0])

  vals = vals[1:]
  names = names[1:]

  vals_adjusted, names_adjusted = calc_f1(labels, adjusted_probs, model_path, name='adjusted_f1')
  mean_vals.append(vals_adjusted[0])
  mean_names.append(names_adjusted[0])
  
  vals += vals_adjusted[1:]
  names += names_adjusted[1:]
  
  vals_loss, names_loss = calc_loss(labels, probs, model_path)

  mean_vals.append(vals_loss[0])
  mean_names.append(names_loss[0])

  vals += vals_loss[1:]
  names += names_loss[1:]
  
  probs = predicts if not FLAGS.auc_need_softmax else probs
  vals_auc, names_auc = calc_auc(labels, probs, model_path)

  mean_vals.append(vals_auc[0])
  mean_names.append(names_auc[0])

  vals += vals_auc[1:]
  names += names_auc[1:] 

  vals = mean_vals + vals 
  names = mean_names + names 

  return vals, names
  
valid_write = None
infer_write = None 

def write(ids, labels, predicts, ofile, ofile2=None, is_infer=False):
  infos = valid_infos if not is_infer else test_infos
  df = pd.DataFrame()
  df['id'] = ids
  contents = [infos[id]['content_str'] for id in ids]
  df['content'] = contents
  if labels is not None:
    for i in range(len(ATTRIBUTES)):
      df[ATTRIBUTES[i] + '_y'] = labels[:,i] - 2
  for i in range(len(ATTRIBUTES)):
    # nitice if na only then if -1 means predict na as finally should be -2
    if FLAGS.loss_type == 'regression':
      df[ATTRIBUTES[i]] = predicts[:,i]
    else:
      df[ATTRIBUTES[i]] = np.argmax(predicts[:,i], 1) - 2

  if is_infer:
    df2 = df.sort_values('id') 
    df2.to_csv(ofile, index=False, encoding="utf_8_sig")
  
  num_classes = NUM_CLASSES if FLAGS.binary_class_index is None else 1
  if FLAGS.loss_type == 'regression':
    num_classes = 1
  df['score'] = [list(x) for x in np.reshape(predicts, [-1, NUM_ATTRIBUTES * num_classes])]
  # for inference using length buckts need to sort, so for safe just all sort
  try:
    ids = [int(x) for x in ids]
    df['id'] = ids
  except Exception:
    pass
  df= df.sort_values('id') 
  if not is_infer:
    # TODO FIXME new run, seg fild seems luanma.. only on p40 new run..
    #df['seg'] = [ids2text.ids2text(infos[str(id)]['content'], sep='|') for id in ids]
    df.to_csv(ofile, index=False, encoding="utf_8_sig")
  if is_infer:
    ## write debug
    #df2 = df
    #df2['seg'] = [ids2text.ids2text(infos[str(id)]['content'], sep='|') for id in ids]
    #df2.to_csv(ofile2, index=False, encoding="utf_8_sig")
    df.to_csv(ofile2, index=False, encoding="utf_8_sig")

def valid_write(ids, labels, predicts, ofile):
  return write(ids, labels, predicts, ofile)

def infer_write(ids, predicts, ofile, ofile2):
  return write(ids, None, predicts, ofile, ofile2, is_infer=True)


def evaluate_file(file):
  print('-------------------------', file)
  df = pd.read_csv(file)
  
  scores = df['score']
  scores = [gezi.str2scores(score) for score in scores] 
  scores = np.array(scores)
  
  predicts = np.reshape(scores, [-1, NUM_ATTRIBUTES, NUM_CLASSES])  
  
  # for auc might need to do this 
  #predicts /= 26
  
  idx = 2
  length = NUM_ATTRIBUTES 

  labels = df.iloc[:,idx:idx+length].values
  labels += 2

  #print(labels.shape, predicts.shape)
  assert labels.shape[0] == 15000, labels.shape[0]
  vals, names = evaluate(labels, predicts)

  if FLAGS.show_detail:
    for name, val in zip(names, vals):
      print(name, val)

  print('---------------------------------')
  for name, val in zip(names, vals):
    if 'mean' in name:
      print(name, val)

  lens = [len(x) for x in df['content'].values]
  predicts1 = []
  predicts2 = []

  labels1 = []
  labels2 = []
  for len_, label, predict in zip(lens, labels, predicts):
    if len_ >= FLAGS.len_thre:
      predicts2.append(predict)
      labels2.append(label)
    else:
      predicts1.append(predict)
      labels1.append(label)
  predicts1 = np.array(predicts1)
  labels1 = np.array(labels1)
  print('num docs len < ', FLAGS.len_thre, len(predicts1))
  vals1, names1 = evaluate(labels1, predicts1)
  for name, val in zip(names1, vals1):
    if 'mean' in name:
      print(name, val) 
  predicts2 = np.array(predicts2)
  labels2 = np.array(labels2) 
  print('num docs len >= ', FLAGS.len_thre, len(predicts2))
  vals2, names2 = evaluate(labels2, predicts2)
  for name, val in zip(names2, vals2):
    if 'mean' in name:
      print(name, val) 

  return vals, names

if __name__ == '__main__':
  load_class_weights()
  input = FLAGS.i
  os.makedirs('./bak', exist_ok=True)
  if os.path.isdir(input):
    df = pd.DataFrame()
    fnames = []
    mnames = []
    m = {}
    for file in glob.glob('%s/*valid.csv' % input):
      try:
        fname = os.path.basename(file)
        fnames.append(fname)
        fname = gezi.strip_suffix(fname, '.valid.csv')

        if 'ensemble' in file:
          mname = file
          suffix = ''
        else: 
          if '_ckpt-' in fname:
            mname, suffix = fname.split('_ckpt-')
          else:
            mname, suffix = fname.split('_model.ckpt-')
            
        mnames.append(mname)
        
        vals, names = evaluate_file(file)
        for val, name in zip(vals, names):
          if name not in m:
            m[name] = [val]
          else:
            m[name].append(val)
          if name == FLAGS.metric_name and (val < FLAGS.min_thre or val > FLAGS.max_thre):
            print('-----remove file', file, '%s:%f' % (FLAGS.metric_name, val))
            command = 'mv %s.* ./bak' % fname
            print(command)
            os.system(command)
      except Exception:
        print(file)
        traceback.print_exc()

    df['model'] = mnames
    df['file'] = fnames
    for key, val in m.items():
      df[key] = val
    df = df.sort_values('adjusted_f1/mean', ascending=False)
    df.to_csv('models.csv', index=False)
  else:
    evaluate_file(FLAGS.i)
 
