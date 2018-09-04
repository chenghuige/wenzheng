#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   algos_factory.py
#        \author   chenghuige  
#          \date   2016-09-17 19:42:42.947589
#   \Description  
# ==============================================================================

"""
Should move to util
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import sys, inspect


from algos.classifier import Classifier
from algos.char_lm import CharLm

class Algos:
  classifier = 'classifier'
  lm = 'lm'

_trainer_map = {
    Algos.classifier: Classifier,
    Algos.lm: CharLm
  }

#TODO this is c++ way, use yaxml python way pass BowTrainer.. like this directly
#'cnn'.. parmas might remove? just as trainer, predictor normal params
# --algo discriminant --encoder bow or --algo discriminant --encoder rnn

def _gen_trainer(algo, is_training=True):
  trainer_map = _trainer_map
  if algo not in trainer_map:
    raise ValueError('Unsupported algo %s'%algo) 
  trainer_fn = trainer_map[algo]
  # if 'encoder_type' in inspect.getargspec(trainer_fn.__init__).args:
  #   return trainer_fn(encoder_type=algo.split('_')[-1], is_training=is_training)
  # else:
  return trainer_fn(is_training=is_training)

def _gen_predictor(algo):
  predictor_map = _trainer_map
  if algo not in predictor_map:
    print('Unsupported algo %s'%algo, file=sys.stderr) 
    return None
  precitor_fn = predictor_map[algo]
  # if 'encoder_type' in inspect.getargspec(precitor_fn.__init__).args:
  #   return precitor_fn(encoder_type=algo.split('_')[-1])
  # else:
  return precitor_fn(is_training=False, is_predict=True)

#TODO use tf.make_template to remove "model_init" scope?
def gen_predictor(algo, reuse=None):
  with tf.variable_scope("init", reuse=reuse):
    predictor = _gen_predictor(algo)
  return predictor
  
def gen_trainer(algo, reuse=None):
  with tf.variable_scope("init", reuse=reuse):
    trainer = _gen_trainer(algo)
  return trainer

def gen_evaluator(algo, reuse=True):
  with tf.variable_scope("init", reuse=reuse):
    evaluator = _gen_trainer(algo, is_training=False)
  return evaluator

def gen_validator(algo, reuse=True):
  with tf.variable_scope("init", reuse=reuse):
    validator = _gen_trainer(algo, is_training=False)
  return validator  

def gen_trainer_and_predictor(algo):
  trainer = gen_tranier(algo, reuse=None)
  predictor = gen_predictor(algo, reuse=True)
  return trainer, predictor

def gen_trainer_and_validator(algo):
  trainer = gen_trainer(algo, reuse=None)
  validator = gen_validator(algo, reuse=True)
  return trainer, validator

def gen_all(algo):
  trainer = gen_trainer(algo, reuse=None)
  validator = gen_validator(algo, reuse=True)
  predictor = gen_predictor(algo, reuse=True)
  return trainer, validator, predictor 

