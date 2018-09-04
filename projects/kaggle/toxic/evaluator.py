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

from collections import defaultdict

import numpy as np

from gezi import Timer
import gezi
import melt 
logging = melt.logging
import algos_factory
import string
import re
from collections import Counter
#from tqdm import tqdm
from sklearn import metrics
import pandas as pd
pd.set_option('display.max_colwidth', -1)

from deepiu.util import ids2text
from algos.config import CLASSES
  
ids_set = set([
'1308c8f69855b3a3',
'28349d1f699f56ab',
'15eb1135c7e08836',
'01fa2f1880fecd30',
'170bccd278616de1',
'1740be70379b74e4',
'171ee792bb73c98b',
'006b888560bcdfcd',
'0c8ecd6066058328',
'1ef097f8808fce67',
'003217c3eb469ba9',
'006b888560bcdfcd',
'00510c3d06745849',
])

def slim_comment(comment, limit=1000):
  if len(comment) <= limit:
    return comment
  else:
    return comment[:900] + '...' + comment[-100:]

def calc_auc(predicts, classes):
  total_auc = 0. 
  aucs = [0.] * len(CLASSES)
  for i, class_ in enumerate(CLASSES):
    fpr, tpr, thresholds = metrics.roc_curve(classes[:, i], predicts[:, i])
    auc = metrics.auc(fpr, tpr)
    aucs[i] = auc
    total_auc += auc
  auc = total_auc / len(CLASSES) 
  return auc, aucs

def write_evaluate(ids, predicts, classes, comment_strs, comment_tokens_strs, model_path, results=None, names=None):
  if results:
    valid_info_file = model_path + '.valid_info'
    with open(valid_info_file, 'w') as out:
      for name, result in zip(names, results):
        print(name, result, file=out)

  result_file = model_path + '.valid'
  print('save to result file', result_file, file=sys.stderr)
  test_predicts = pd.DataFrame(data=predicts, columns=CLASSES)
  test_predicts["id"] = ids
  test_predicts = test_predicts[["id"] + CLASSES]
  test_predicts.to_csv(result_file, index=False)

  debug_result_file = model_path + '.valid_debug'
  test_predicts["comment"] = comment_strs
  test_predicts["comment_tokens"] = comment_tokens_strs
  test_predicts = test_predicts[["id"] + CLASSES + ["comment", "comment_tokens"]]
  
  truth = pd.DataFrame(data=classes, columns=CLASSES)
  truth["id"] = ids
  truth = truth[["id"] + CLASSES]

  valid_truth = test_predicts.join(truth, lsuffix='_p')
  valid_truth = valid_truth[["id"] + CLASSES + [ x +'_p' for x in CLASSES] + ["comment", "comment_tokens"]]
  valid_truth = valid_truth.sort_values(['toxic_p'], ascending=[0])
  valid_truth.to_csv(debug_result_file, index=False)
  
  for class_ in CLASSES:
    if class_ != 'toxic':
      valid_truth = valid_truth.sort_values([class_ + '_p'], ascending=[0])  
    sort_file = model_path + '.valid_sort_%s' % class_
    valid_sort = valid_truth[["id"] + CLASSES + ["comment"]][:1000]
    valid_sort.to_csv(sort_file, index=False)  
    valid_bad_file = model_path + '.valid_bad_%s' % class_
    num = 100
    with open(valid_bad_file, 'w') as out:
      name = class_
      df = valid_truth[['id', name, name + '_p', 'comment', 'comment_tokens']]
      num_norecall = 0
      print('%s no recall' % name, file=out)
      for i in reversed(range(len(df))):
        if df.iloc[i][name] == 1.:
          print(i, df.iloc[[i]], file=out)
          if num_norecall < 2:
            if FLAGS.log_level > 1:
              logging.info('norecall', name, i, df.iloc[i]['id'], "{0:0.3f}".format(df.iloc[i][name + '_p']), slim_comment(df.iloc[i]['comment']))
              logging.info('norecall', name, i, df.iloc[i]['id'], "{0:0.3f}".format(df.iloc[i][name + '_p']), df.iloc[i]['comment_tokens'][:500])
          num_norecall += 1
          if num_norecall == num:
            break

      print('%s wrong' % name, file=out)
      num_wrong = 0
      for i in range(len(df)):
        if df.iloc[i][name] == 0.:
          print(i, df.iloc[[i]], file=out)
          if num_wrong < 1:
            if FLAGS.log_level > 1:
              logging.info('wrong', name, i, df.iloc[i]['id'], "{0:0.3f}".format(df.iloc[i][name + '_p']), slim_comment(df.iloc[i]['comment']))
              logging.info('wrong', name, i, df.iloc[i]['id'], "{0:0.3f}".format(df.iloc[i][name + '_p']), df.iloc[i]['comment_tokens'][:500])
          num_wrong += 1
          if num_wrong == num:
            break

def write_inference(ids, predicts, comment_strs, comment_tokens_strs, model_path):
  result_file = model_path + '.infer'
  print('save to result file', result_file, file=sys.stderr)
  test_predicts = pd.DataFrame(data=predicts, columns=CLASSES)
  test_predicts["id"] = ids
  test_predicts = test_predicts[["id"] + CLASSES]
  test_predicts.to_csv(result_file, index=False)

  debug_result_file = model_path + '.infer_debug'
  test_predicts["comment"] = comment_strs
  test_predicts["comment_tokens"] = comment_tokens_strs
  test_predicts = test_predicts.sort_values(['toxic'], ascending=[0])
  test_predicts.to_csv(debug_result_file, index=False)

  for class_ in CLASSES:
    if class_ != 'toxic':
      test_predicts = test_predicts.nlargest(1000, [class_])
    sort_file = model_path + '.infer_sort_%s' % class_
    infer_sort = test_predicts[["id", "comment"]][:1000]
    infer_sort.to_csv(sort_file, index=False)


decay = None

def evaluate(eval_ops, iterator, validator, model_path=None, sess=None):
  ids2text.init()
  global ids_set
  if os.path.exists('ids.txt'):
    ids_set = set()
    for line in open('ids.txt'):
      ids_set.add(line.strip())

  if model_path:
    ids_list = []
    comment_strs_list = []
    comment_tokens_strs_list = []
  predictions_list = []
  classes_list = []
  losses = []
  class_losses = []

  m = {}

  loss_, predictions_, class_loss_, words_scores_, result = eval_ops

  if not sess:
    sess = melt.get_session() 
  sess.run(iterator.initializer)
  
  # TODO using tf.contrib.metric steaming_auc ingraph evaluate?

  # debug 
  # print('-------------------finetune emb:', sess.run(validator.finetune_emb))
  # print('-------------------global step:', sess.run(validator.global_step))

  if FLAGS.finetune_emb_step:
    print('finetune emb:', sess.run(validator.finetune_emb))

  try:
    num_candidates = 0
    while True:
      ids, loss, predictions, class_loss, words_scores, classes, comment_strs, comments = \
          sess.run([result.id, loss_, predictions_, class_loss_, words_scores_, result.classes, result.comment_str, result.comment])

      ids = gezi.decode(ids)
      comment_strs = gezi.decode(comment_strs)

      losses.append(loss)
      class_losses.append(class_loss)
      if model_path:
        ids_list.append(ids)
        comment_strs_list.append(comment_strs)
        comment_tokens_strs = np.array(ids2text.idslist2texts(comments, sep='|'))
        comment_tokens_strs_list.append(comment_tokens_strs)
      predictions_list.append(predictions)
      classes_list.append(classes)

      if FLAGS.log_level >= 1:
        def is_candidate(id, prediction, labels, num_candidates):
          if num_candidates > 2:
            return False
          idx = np.random.randint(len(CLASSES))
          prob = np.random.random()
          #print('-------------', id, idx, labels[idx], prediction[idx])
          if labels[idx] == 1. and prediction[idx] < 0.5:
            if (1 - prediction[idx]) > prob:
              return True
          else:
            if prediction[idx] > 0.5 and prediction[idx] > prob:
              return True
          return False

        # TODO better 
        if words_scores is not None:
          for id, prediction, words_score, class_, comment_str, comment in zip(ids, predictions, words_scores, classes, comment_strs, comments):
            if is_candidate(id, prediction, class_, num_candidates) or id in ids_set:
              if not id in ids_set:
                num_candidates += 1
              logging.info(id, list(zip(class_, map("{:.2f}".format, prediction))), slim_comment(comment_str))
              comment_tokens = [ids2text.vocab.key(x) for x in comment if x > 0]
              total_score = sum(words_score)
              words_score = [x / total_score * len(comment_tokens) for x in words_score]
              words_score = words_score[: len(comment_tokens)]
              #words_score = map("{:0.1f}".format, words_score)
              words_importance = list(zip(comment_tokens, words_score))
              #logging.info(id, words_importance)
              words_importance.sort(key=lambda x: x[1], reverse=True)
              logging.info(id, [(x, "{:.2f}".format(y)) for x, y in words_importance[:6]])
        else:
          for id, prediction, class_, comment_str, comment in zip(ids, predictions, classes, comment_strs, comments):
            if is_candidate(id, prediction, class_, num_candidates) or id in ids_set: 
              if not id in ids_set:
                num_candidates += 1
              logging.info(id, list(zip(class_, map("{:.2f}".format, prediction))), slim_comment(comment_str))
              comment_tokens = [ids2text.vocab.key(x) for x in comment if x > 0]
              logging.info(id, comment_tokens)
  except tf.errors.OutOfRangeError:
    loss = np.mean(losses)
    class_loss = np.mean(class_losses, 0)
    predicts = np.concatenate(predictions_list)
    classes = np.concatenate(classes_list)
    auc, aucs = calc_auc(predicts, classes)

    # not model_path for we do not want to decay in epoch evaluate step
    weights = None
    if FLAGS.decay_target and not model_path:
      decay_target = FLAGS.decay_target
      logging.info('decay_target:', decay_target)
      global decay 
      from util import WeightsDecay
      if decay is None:
        cmp = 'less' if decay_target == 'loss' else 'greater'
        #decay_factor = [FLAGS.decay_factor] * len(CLASSES)
        #decay_factor[0] = 0.98
        #decay_factor[1] = 0.98
        decay = WeightsDecay('lr_ratios', names=CLASSES, patience=FLAGS.decay_patience, decay=FLAGS.decay_factor, cmp=cmp)
        
      if decay_target == 'loss':
        weights = decay.add(class_loss)
      else:
        weights = decay.add(aucs)

    results = [loss, auc] + aucs + list(class_loss)
    names = ['metric/valid/loss/avg', 'metric/valid/auc/avg'] \
            + ['metric/valid/auc/' + x for x in CLASSES] \
            + ['metric/valid/loss/' + x for x in CLASSES]
    
    if weights is not None:
      results += list(weights)
      names += ['weights/' + x for x in CLASSES]

    if model_path:
      write_evaluate(np.concatenate(ids_list), predicts, classes, 
                     np.concatenate(comment_strs_list), np.concatenate(comment_tokens_strs_list),
                     model_path, results, names)

    return results, names

def inference(inference_ops, iterator, model_path, sess=None):
  predictions_, result = inference_ops
  
  ids_list = []
  comment_strs_list = []
  comment_tokens_strs_list = []
  predictions_list = []
  
  timer = gezi.Timer('%s inference' % model_path)
  
  if not sess:
    sess = melt.get_session() 
  sess.run(iterator.initializer)
  try:
    while True:
      ids, predictions, comment_strs, comment_tokens_strs, comments = sess.run([result.id, predictions_, result.comment_str, result.comment_tokens_str, result.comment])
      
      ids = gezi.decode(ids)
      comment_strs = gezi.decode(comment_strs)
      comment_tokens_strs = gezi.decode(comment_tokens_strs)

      ids_list.append(ids)
      comment_strs_list.append(comment_strs)
      #comment_tokens_strs_list.append(comment_tokens_strs)
      comment_tokens_strs = np.array(ids2text.idslist2texts(comments, sep='|'))
      comment_tokens_strs_list.append(comment_tokens_strs)
      predictions_list.append(predictions)
      if len(ids_list) % 100 == 0:
        print('inference %d' % len(ids_list), end='\r', file=sys.stderr)
  except tf.errors.OutOfRangeError:
    predicts = np.concatenate(predictions_list)
    print('finish run inference %d' % len(predicts), file=sys.stderr)

    write_inference(np.concatenate(ids_list), predicts,  
                    np.concatenate(comment_strs_list), np.concatenate(comment_tokens_strs_list),
                    model_path)
  timer.print_elapsed()
