#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   predictor.py
#        \author   chenghuige  
#          \date   2016-09-30 15:19:35.984852
#   \Description  
# ==============================================================================
"""
This predictor will read from checkpoint and meta graph, only depend on tensorflow
no other code dpendences, so can help hadoop or online deploy,
and also can run inference without code of building net/graph

TODO test use tf.Session() instead of melt.get_session()
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python import debug as tf_debug

import os, sys, math
import numpy as np
import gezi
import melt 

import operator 
import traceback

def get_model_dir_and_path(model_dir, model_name=None):
  model_path = model_dir
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    #model_path = '%s/%s'%(model_dir, os.path.basename(ckpt.model_checkpoint_path)) 
    model_path = os.path.join(model_dir, os.path.basename(ckpt.model_checkpoint_path))
  else:
    model_path = model_dir if model_name is None else os.path.join(model_dir, model_name)
  #if not os.path.exists(model_path):
  #  raise ValueError(model_path)
  return os.path.dirname(model_path), model_path

# tf.get_default_graph().get_all_collection_keys() get all keys
# graph.get_all_collection_keys
# TODO FIXME even without / might still be tensor key name not collection key name
def get_tensor_from_key(key, graph, index=-1):
  if isinstance(key, str):
    if not '/' in key:
      try:
        ops = graph.get_collection(key)
        if len(ops) > 1:
          #print('Warning: ops more then 1 for {}, ops:{}, index:{}'.format(key, ops, index))
          pass
        return ops[index]
      except Exception:
        print
    else:
      if not key.endswith(':0'):
        key = key + ':0'
      #print('------------', [n.name for n in graph.as_graph_def().node])
      try:
        op = graph.get_tensor_by_name(key)
        return op
      except Exception:
        #print(traceback.format_exc())
        key = 'prefix/' + key 
        op = graph.get_tensor_by_name(key)
        return op
  else:
    return key

class Predictor(object):
  def __init__(self, model_dir=None, meta_graph=None, model_name=None, 
               debug=False, sess=None, graph=None,
               frozen_graph=None, frozen_graph_name='prefix',
               no_frozen_graph=False,
               random_seed=1234):
    super(Predictor, self).__init__()
    self.sess = sess
    if self.sess is None:
      ##---TODO tf.Session() if sess is None
      #self.sess = tf.InteractiveSession()
      #self.sess = melt.get_session() #make sure use one same global/share sess in your graph
      self.graph = graph or tf.Graph()
      self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=self.graph) #by default to use new Session, so not conflict with previous Predictors(like overwide values)
      if debug:
        self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
    #ops will be map and internal list like
    #{ text : [op1, op2], text2 : [op3, op4, op5], text3 : [op6] }
    
    if random_seed is not None:
      tf.set_random_seed(random_seed)

    self.frozen_graph_name = frozen_graph_name
    self.no_frozen_graph = no_frozen_graph
    if frozen_graph is None:
      if model_dir is not None:
        self.restore(model_dir, meta_graph, model_name)
    else:
      self.load_graph(frozen_graph, frozen_graph_name)

  #by default will use last one
  def inference(self, key, feed_dict=None, index=-1):
    if not isinstance(key, (list, tuple)):
      return self.sess.run(get_tensor_from_key(key, self.graph, index), feed_dict=feed_dict)
    else:
      keys = key 
      if not isinstance(index, (list, tuple)):
        indexes = [index] * len(keys)
      else:
        indexes = index 
      keys = [get_tensor_from_key(key, self.graph, index) for key,index in zip(keys, indexes)]
      return self.sess.run(keys, feed_dict=feed_dict)

  def predict(self, key, feed_dict=None, index=-1):
    return self.inference(key, feed_dict, index)

  def run(self, key, feed_dict=None):
    return self.sess.run(key, feed_dict)

  def restore(self, model_dir, meta_graph=None, model_name=None, random_seed=None):
    """
    do not need to create graph
    restore graph from meta file then restore values from checkpoint file
    """
    model_dir, model_path = get_model_dir_and_path(model_dir, model_name)
    self.model_path = model_path

    frozen_graph_file = '%s.pb' % model_path
    if os.path.exists(frozen_graph_file) and not self.no_frozen_graph:
      frozen_map_file = '%s.map' % model_path
      return self.load_graph(frozen_graph_file, self.frozen_graph_name, frozen_map_file=frozen_map_file)

    timer = gezi.Timer('restore meta grpah and model ok %s' % model_path)
    if meta_graph is None:
      meta_graph = '%s.meta' % model_path
    ##https://github.com/tensorflow/tensorflow/issues/4603
    #https://stackoverflow.com/questions/37649060/tensorflow-restoring-a-graph-and-model-then-running-evaluation-on-a-single-imag
    with self.sess.graph.as_default():
      saver = tf.train.import_meta_graph(meta_graph)
      saver.restore(self.sess, model_path)
    if random_seed is not None:
      tf.set_random_seed(random_seed)

    #---so maybe do not use num_epochs or not save num_epochs variable!!!! can set but input producer not use, stop by my flow loop
    #---TODO not work remove can run but hang  FIXME add predictor + exact_predictor during train will face
    #@gauravsindhwani , can you still run the code successfully after you remove these two collections since they are actually part of the graph. 
    #I try your way but find the program is stuck after restoring."
    #https://github.com/tensorflow/tensorflow/issues/9747
    #tf.get_default_graph().clear_collection("queue_runners")
    #tf.get_default_graph().clear_collection("local_variables")
    #--for num_epochs not 0
    #tf.get_default_graph().clear_collection("local_variables")
    #self.sess.run(tf.local_variables_initializer())

    #https://stackoverflow.com/questions/44251666/how-to-initialize-tensorflow-variable-that-wasnt-saved-other-than-with-tf-globa
    #melt.initialize_uninitialized_vars(self.sess)

    timer.print_elapsed()
    return self.sess
  #http://ndres.me/post/convert-caffe-to-tensorflow/
  def load_graph(self, frozen_graph_file, frozen_graph_name='prefix', frozen_map_file=None):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    timer = gezi.Timer('load frozen graph from %s with mapfile %s' % (frozen_graph_file, frozen_map_file))
    with tf.gfile.GFile(frozen_graph_file, "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with self.sess.graph.as_default() as graph:
      tf.import_graph_def(
        graph_def,
        input_map=None,
        return_elements=None,
        name=frozen_graph_name,
        #op_dict=None,
        producer_op_list=None
      )

      if frozen_map_file is not None and os.path.exists(frozen_map_file):
        for line in open(frozen_map_file):
          cname, key = line.strip().split('\t')
          if not (key.endswith(':0') or key.endswith(':1') or key.endswith(':2')):
            key = '%s:0' % key
          tensor = graph.get_tensor_by_name('%s/%s' % (frozen_graph_name, key))
          graph.add_to_collection(cname, tensor)

    timer.print_elapsed()
    return graph

class SimplePredictor(object):
  def __init__(self, 
              model_dir, 
              key=None, 
              feed=None,
              index=0,
              meta_graph=None, 
              model_name=None, 
              debug=False, 
              sess=None,
              graph=None):
    self.predictor = Predictor(model_dir, meta_graph, model_name, debug, sess, graph)
    self.graph = self.predictor.graph
    key = key or 'predictions'
    feed = feed or 'feed'
    try:
      self.key = self.graph.get_collection(key)[index] 
    except Exception:
      self.key = None
  
    self.index = index
    try:
      self.feed = self.graph.get_collection(feed)[index]
    except Exception:
      self.feed = None

    self.sess = self.predictor.sess

  def inference(self, input, key=None, index=None, feed=None):
    if key is None:
      key = self.key
    if index is None:
      index = self.index
    if feed is None:
      feed = self.feed 
    else:
      feed = self.graph.get_collection(feed)[index]

    feed_dict = {
        feed: input
    }
    
    return self.predictor.inference(key, feed_dict=feed_dict, index=index)

#TODO lfeed, rfeed .. should be named as lfeed, rfeed
class SimPredictor(object):
  def __init__(self, 
              model_dir, 
              key=None, 
              lfeed=None,
              rfeed=None,
              index=0,
              meta_graph=None, 
              model_name=None, 
              debug=False, 
              sess=None,
              graph=None):
    self.predictor = Predictor(model_dir, meta_graph, model_name, debug, sess, graph)
    self.graph = self.predictor.graph
    key = key or 'score'
    lfeed = lfeed or 'lfeed'
    rfeed = rfeed or 'rfeed'
    #print(self.graph.get_all_collection_keys())
    self.key = self.graph.get_collection(key)[index] 
    self.index = index
    self.lfeed = self.graph.get_collection(lfeed)[index]
    self.rfeed = self.graph.get_collection(rfeed)[index]

    self.sess = self.predictor.sess

  def inference(self, ltext, rtext=None, key=None, index=None, lfeed=None, rfeed=None):
    if key is None:
      key = self.key
    if index is None:
      index = self.index
    if lfeed is None:
      lfeed = self.lfeed 
    else:
      lfeed = self.graph.get_collection(lfeed)[index]
    if rfeed is None:
      rfeed = self.rfeed
    else:
      rfeed = self.graph.get_collection(rfeed)[index]

    if rtext is not None:
      feed_dict = {
        lfeed: ltext,
        rfeed: rtext
      }
      return self.predictor.inference(key, feed_dict=feed_dict, index=index)
    else:
      feed_dict = {
        lfeed: ltext
      }

    return self.predictor.inference(key, feed_dict=feed_dict, index=index)


  def predict(self, ltext, rtext=None, key=None, index=None):
    return self.inference(ltext, rtext, key, index)

  def elementwise_predict(self, ltexts, rtexts, expand_left=True):
    scores = []
    #if len(rtexts) >= len(ltexts):
    if expand_left:
      for ltext in ltexts:
        stacked_ltexts = np.array([ltext] * len(rtexts))
        score = self.predict(stacked_ltexts, rtexts)
        score = np.squeeze(score) 
        scores.append(score)
    else:
      for rtext in rtexts:
        stacked_rtexts = np.array([rtext] * len(ltexts))
        score = self.predict(ltexts, stacked_rtexts)
        score = np.squeeze(score) 
        scores.append(score)
    return np.array(scores) 

  def onebyone_predict(self, ltexts, rtexts_list):
    scores = []
    for ltext, rtexts in zip(ltexts, rtexts_list):
      score = self.predict([ltext], rtexts)
      score = np.squeeze(score)
      scores.append(score)
    return np.array(scores)

  #def bulk_predict(self, ltexts, rtexts_list, buffer_size=512):
  def bulk_predict(self, ltexts, rtexts_list, buffer_size=50):
    """
    ltexts [batch_size,..]
    rtexts [batch_size, num_texts,..]
    TODO support split batch, so can avoid gpu mem issue
    """
    stacked_ltexts = []
    stacked_rtexts = []
    boundaries = []
    for ltext, rtexts in zip(ltexts, rtexts_list):
      stacked_ltexts += [ltext] * len(rtexts)
      stacked_rtexts += list(rtexts)
      boundaries.append(len(rtexts))

    stacked_ltexts = np.array(stacked_ltexts)
    stacked_rtexts = np.array(stacked_rtexts)

    if len(stacked_ltexts) <= buffer_size:
      #[n, 1] <- [n, 1] [n, 1]
      score = self.predict(stacked_ltexts, stacked_rtexts)
    else:
      start = 0
      score_list = []
      while start < len(stacked_ltexts):
        end = start + buffer_size
        score = self.predict(stacked_ltexts[start: end], stacked_rtexts[start: end])
        score_list += list(score)
        start += buffer_size
      score = np.array(score_list)
    score = np.squeeze(score)

    results_list = []
    index = 0
    for num_rtexts in boundaries:
      results = []
      for i in range(num_rtexts):
        results.append(score[index])
        index += 1
      results_list.append(np.array(results))
    results_list = np.array(results_list)
    return results_list

  def top_k(self, ltext, rtext, k=1, key=None):
    feed_dict = {
      self.lfeed: ltext,
      self.rfeed: rtext
    }
    if key is None:
      key = 'nearby'
    try:
      values, indices = self.predictor.inference(key, feed_dict=feed_dict, index=self.index)
      return values[:k], indices[:k]
    except Exception:
      # score = self.predict(ltext, rtext)
      # indices = (-score).argsort()[:k]
      # # result
      # x = arr.shape[0]
      # #like [0, 0, 1, 1] [1, 0, 0, 1] ->...  choose (0,1), (0, 0), (1,0), (1, 1)
      # values = score[np.repeat(np.arange(x), N), indices.ravel()].reshape(x, k)
      # return values, indices
      scores = tf.get_collection(self.key)[self.index]
      vals, indexes = tf.nn.top_k(scores, k)
      return self.sess.run([vals, indexes], feed_dict=feed_dict)

#different session for predictor and exact_predictor all using index 0! if work not correclty try to change Predictor default behave use melt.get_session() TODO
class RerankSimPredictor(object):
  def __init__(self, model_dir, exact_model_dir, num_rerank=100, 
              lfeed=None, rfeed=None, exact_lfeed=None, exact_rfeed=None, 
              key=None, exact_key=None, index=0, exact_index=0, sess=None, exact_sess=None, graph=None, exact_graph=None):
    self.predictor = SimPredictor(model_dir, index=index, lfeed=lfeed, rfeed=rfeed, key=key, sess=sess, graph=graph)
    #TODO FIXME for safe use -1, should be 1 also ok, but not sure why dual_bow has two 'score'.. 
    #[<tf.Tensor 'dual_bow/main/dual_textsim_1/dot/MatMul:0' shape=(?, ?) dtype=float32>, <tf.Tensor 'dual_bow/main/dual_textsim_1/dot/MatMul:0' shape=(?, ?) dtype=float32>,
    # <tf.Tensor 'seq2seq/main/Exp_4:0' shape=(?, 1) dtype=float32>]
    #this is becasue you use evaluator(predictor + exact_predictor) when train seq2seq, so load dual_bow will add one score..
    self.exact_predictor = SimPredictor(exact_model_dir, index=exact_index, lfeed=exact_lfeed, rfeed=exact_rfeed, key=exact_key, 
                                        sess=exact_sess, graph=exact_graph)

    self.num_rerank = num_rerank

  def inference(self, ltext, rtext, ratio=1.):
    scores = self.predictor.inference(ltext, rtext)
    if not ratio:
      return scores

    exact_scores = []
    for i, score in enumerate(scores):
      index= (-score).argsort()
      top_index = index[:self.num_rerank]
      exact_rtext = rtext[top_index]
      exact_score = self.exact_predictor.elementwise_predict([ltext[i]], rtext)
      exact_score = np.squeeze(exact_score)
      if ratio < 1.:
        for j in range(len(top_index)):
          exact_score[j] = ratio * exact_score[j] + (1. - ratio) * score[top_index[j]]

      exact_scores.append(exact_score)
    return np.array(exact_score)

  def predict(self, ltext, rtext, ratio=1.):
    return self.predict(ltext, rtext, ratio)

  #TODO do numpy has top_k ? seems argpartition will get topn but not in order
  def top_k(self, ltext, rtext, k=1, ratio=1., sorted=True):
    assert k <= self.num_rerank
    #TODO speed hurt?
    ltext = np.array(ltext)
    rtext = np.array(rtext)
    scores = self.predictor.predict(ltext, rtext)
    
    top_values = []
    top_indices = []

    if not ratio:
      for i, score in enumerate(scores):
        index = (-score).argsort()
        top_values.append(score[index[:k]])
        top_indices.append(index[:k])
      return np.array(top_values), np.array(top_indices)

    for i, score in enumerate(scores):
      index = (-score).argsort()
      print(index,  np.argpartition(-score, self.num_rerank))
      if ratio:
        top_index = index[:self.num_rerank]
        exact_rtext = rtext[top_index]
        exact_score = self.exact_predictor.elementwise_predict([ltext[i]], exact_rtext)
        exact_score = np.squeeze(exact_score)
        if ratio < 1.:
          for j in range(len(top_index)):
            exact_score[j] = ratio * exact_score[j] + (1. - ratio) * score[top_index[j]]

        exact_index = (-exact_score).argsort()

        new_index = [x for x in index]
        for j in range(len(exact_index)):
          new_index[j] = index[exact_index[j]]
        index = new_index
      
      top_values.append(exact_score[exact_index[:k]])
      top_indices.append(index[:k])

    return np.array(top_values), np.array(top_indices)

class WordsImportancePredictor(object):
  def __init__(self, model_dir, key=None, feed=None, index=0, sess=None):
    self.predictor = Predictor(model_dir, sess=sess)
    self.graph = self.predictor.graph
    self.index = index 

    if key is None:
      self.key = self.graph.get_collection('words_importance')[index]
    else:
      self.key = key

    if feed is None:
      self.feed = self.graph.get_collection('rfeed')[index]
    else:
      self.feed = feed


  def inference(self, inputs):
    feed_dict = {self.feed: inputs}
    return self.predictor.inference(self.key, feed_dict=feed_dict, index=self.index)

  def predict(self, inputs):
    return self.inference(inputs)

class TextPredictor(object):
  def __init__(self, 
              model_dir, 
              feed=None,
              text_key='beam_text',
              score_key='beam_text_score',
              lfeed=None,
              rfeed=None,
              key=None,
              index=0,
              meta_graph=None, 
              model_name=None,
              current_length_normalization_factor=None,
              length_normalization_fator=None, 
              vocab=None,
              debug=False, 
              sess=None,
              graph=None):
    self.predictor = SimPredictor(model_dir, index=index, lfeed=lfeed, rfeed=rfeed, key=key, sess=sess, graph=graph)
    self.ori_predictor = self.predictor.predictor
    self.graph = self.predictor.graph
    self.index = index

    self.text_key = text_key
    self.score_key = score_key

    self.current_length_normalization_factor = current_length_normalization_factor
    self.length_normalization_fator = length_normalization_fator
    self.vocab = vocab

    assert self.vocab or (self.current_length_normalization_factor == self.length_normalization_fator)

    if feed is None:
      try:
        self.feed = self.graph.get_collection('feed')[index]
      except Exception:
        self.feed = self.graph.get_collection('lfeed')[index]
    else:
      self.feed = feed

  def _get_text_len(self, text):
    len_ = 0
    for wid in text:
      len_ += 1
      if wid == self.vocab.end_id():
        break 
    return len_

  def _get_texts_len(self, texts):
    lens_ = [self._get_text_len(text) for text in texts]
    return np.array(lens_)

  def inference(self, inputs, text_key=None, score_key=None, index=None):
    if text_key is None:
      text_key = self.text_key

    if score_key is None:
      score_key = self.score_key

    if index is None:
      index = self.index

    feed_dict = {
      self.feed: inputs
    }

    texts, scores = self.ori_predictor.inference([text_key, score_key], feed_dict=feed_dict, index=index)

    #not used much.. currenlty not used
    if self.current_length_normalization_factor and self.length_normalization_fator and \
         self.current_length_normalization_factor != self.length_normalization_fator:
      sequence_length = self._get_text_len(texts)
      current_normalize_factor = tf.pow(sequence_length, self.current_normalize_factor)
      normalize_factor = tf.pow(sequence_length, self.length_normalization_fator)

      logprobs = math.log(scores)
      logprobs = logprobs * current_normalize_factor / normalize_factor
      scores = math.exp(logprobs)

    return texts, scores

  def predict(self, ltext, rtext, index=-1):
    return self.predictor.predict(ltext, rtext, index=index)

  def elementwise_predict(self, ltexts, rtexts):
    return self.predictor.elementwise_predict(ltexts, rtexts)

  def bulk_predict(self, ltexts, rtexts):
    return self.predictor.bulk_predict(ltexts, rtexts)

  def predict_text(self, inputs, text_key=None, score_key=None, index=None):
    return self.inference(inputs, text_key, score_key, index)

  def predict_full_text(self, inputs, index=None):
    return self.inference(inputs, 'full_text', 'full_text_score', index=index)

class EnsembleTextPredictor(object):
  """
  might has memory issue since, multiple model each with image model inside... TODO
  one way is juse use nomral 1 TextPredictor and each write a file and deal with all result files merge 
  result as ensemble result
  """
  def __init__(self, 
              model_dirs, 
              feed=None,
              text_key='beam_text',
              score_key='beam_text_score',
              current_length_normalization_factor=1.,
              length_normalization_fator=1.,
              vocab=None,
              index=0,
              sess=None):
    if isinstance(model_dirs, str):
      if ',' in model_dirs:
        model_dirs = model_dirs.split(',')
      else:
        model_dirs = [model_dirs]
    self.predictors = [TextPredictor(model_dir, 
                                    feed=feed, 
                                    text_key=text_key, 
                                    score_key=score_key, 
                                    current_length_normalization_factor=current_length_normalization_factor,
                                    length_normalization_fator=length_normalization_fator,
                                    vocab=vocab,
                                    index=index,
                                    sess=sess) \
                       for model_dir in model_dirs]

  #Not correct need to FIMXE one input image has some candidates to calc(different num for different image)
  #need to pack calc and reassign back correctly
  #well below to complex TODO... now just using simple method
  # def inference(self, inputs):
  #   text_map = {}
  #   img2text = {}
  #   for i, predictor in enumerate(self.predictors):
  #     texts, scores = predictor.inference(inputs)
  #     print('----', texts.shape, scores.shape)
  #     for j, text in enumerate(texts):
  #       #HACK TODO now beam search with input TEXT_MAX_WORDS will predict text with length TEXT_MAX_WORDS + 2
  #       #should move the convertion code outside maybe change beam search code is best way
  #       #HACK 9 for ai-chanllenger end_id
  #       text = text[0][:-2]
  #       text = np.array([x if x != 9 else 0 for x in text])
  #       text_str = '\t'.join([str(x) for x in text])
  #       img2text.setdefault(i, set())
  #       img2text[j].add(text_str)
  # #       text_map[text_str] = text


  #   texts = np.array([y for (x, y) in text_map.items()])

  #   print('--------', texts, inputs[0].shape, texts.shape)
  #   scores_list =[predictor.elementwise_predict(inputs, texts) for predictor in self.predictors]

  #   scorer = gezi.AvgScore()
  #   for scores in scores_list:
  #     scorer.add(scores)
  #   scores = scorer.avg_score()
  #   scores = np.array(scores)
  #   print(texts, scores, texts.shape, scores.shape)

  #   results = sorted([(score, text) for score, text in zip(scores, texts)], reverse=True)
  #   #https://stackoverflow.com/questions/12974474/how-to-unzip-a-list-of-tuples-into-individual-lists
  #   scores, texts = zip(*results)

  #   return np.array(texts), np.array(scores)

  def inference(self, inputs):
    ##---below is wrong.. all {} will point to one postion...
    #results = [{}] * len(inputs)
    results = [{} for _ in range(len(inputs))]
    for predictor in self.predictors:
      #[batch_size, beam_size, text]
      texts_list, scores_list = predictor.inference(inputs)
      print('texts_list.shape', texts_list.shape, len(texts_list))
      for i, (texts, scores) in enumerate(zip(texts_list, scores_list)):
        for text, score in zip(texts, scores):
          text_str = '\t'.join([str(x) for x in text])
          results[i].setdefault(text_str, 0.)
          results[i][text_str] += score 

    texts = []
    scores = []
    for result in results:
      sorted_result = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
      texts.append([map(int, sorted_result[0][0].split('\t'))])
      scores.append([sorted_result[0][1]])

    return np.array(texts), np.array(scores)


  def predict(self, inputs):
    return self.inference(inputs)

  def predict_text(self, inputs):
    return self.inference(inputs)
