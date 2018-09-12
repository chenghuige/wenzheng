#!/usr/bin/env python
# ==============================================================================
#          \file   train_tfrecord.py
#        \author   chenghuige  
#          \date   2016-08-16 13:00:55.055143
#   \Description  
# ==============================================================================

"""
@TODO train_tfrecord.py test_tfrecord.py should be rename to train.py test.py
since the work flow works for tfrecord or simple feeding data flow
train.py test.py right now should be rename to train_once.py test_once.py
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import gezi
from gezi import Timer
import melt

from melt.flow.train_once import train_once
from melt.flow.flow import tf_flow
from melt.flow.flow import tf_train_flow

#from melt.util import print_results
#@TODO inside melt can not use melt.print_reults must use melt.util.print_results
#@TODO too many args, using **kwarg ?
def simple_train_flow(ops, 
                      names=None, 
                      gen_feed_dict_fn=None, 
                      deal_results_fn=melt.print_results, 
                      interval_steps=100, 
                      eval_ops=None, 
                      eval_names=None, 
                      gen_eval_feed_dict_fn=None, 
                      deal_eval_results_fn=melt.print_results, 
                      eval_interval_steps=100, 
                      print_time=True, 
                      print_avg_loss=True, 
                      log_dir=None, 
                      num_steps=None, 
                      num_steps_per_epoch=None,
                      metric_eval_fn=None,
                      metric_eval_interval_steps=0,
                      sess=None):
  """
  simple train flow for tr records, without model saving 
  just for test purpose, see examples/sparse-tensor-classification/train-melt.py
  NOTICE: first ops must be train_op(optimizer related) which will later ignored
  """
  print('Will not save model')
  if log_dir:
    print('Will save log to %s'%log_dir)
  else:
    print('Will not save log')
  def train_once_(sess, step):
    train_once(sess, 
               step, 
               ops, 
               names, 
               gen_feed_dict_fn, 
               deal_results_fn, 
               interval_steps, 
               eval_ops, 
               eval_names, 
               gen_eval_feed_dict_fn, 
               deal_eval_results_fn, 
               eval_interval_steps,
               print_time, 
               print_avg_loss, 
               log_dir=log_dir,
               num_steps_per_epoch=num_steps_per_epoch,
               metric_eval_fn=metric_eval_fn,
               metric_eval_interval_steps=metric_eval_interval_steps)
  
  tf_flow(train_once_, num_steps, sess=sess)

def train_flow(ops, 
               names=None, 
               gen_feed_dict_fn=None, 
               deal_results_fn=melt.print_results, 
               interval_steps=100, 
               eval_ops=None, 
               eval_names=None, 
               gen_eval_feed_dict_fn=None, 
               deal_eval_results_fn=melt.print_results, 
               eval_interval_steps=100, 
               eval_loops=1,
               print_time=True, 
               print_avg_loss=True, 
               model_dir=None, 
               max_models_keep=5, 
               save_interval_seconds=600, 
               save_interval_steps=1000,
               freeze_graph=False,
               log_dir=None, 
               no_log=False, 
               num_epochs=None,
               num_steps=None, 
               num_steps_per_epoch=None, 
               optimizer=None, 
               learning_rate=None,
               learning_rate_patience=None,
               learning_rate_decay_factor=None,
               save_model=True, 
               save_interval_epochs=None,
               add_train_var_histogram=False,
               restore_from_latest=True,
               metric_eval_fn=None,
               metric_eval_interval_steps=0,
               valid_interval_epochs=0,
               inference_fn=None, 
               inference_interval_epochs=0,
               summary_excls=None,
               init_fn=None,
               restore_fn=None,
               restore_include=None,
               restore_exclude=None,
               save_all_scope=False,
               variables_to_restore=None,
               variables_to_save=None,
               output_collection_names=None, 
               output_node_names=None,
               write_during_train=True,
               model=None,
               sess=None):
  """
  train flow for tr records, with model saving/reload and summary considered
  summary logs will also write to model_dir 
  see examples/sparse-tensor-classification/train-melt-savemodel.py
  NOTICE: first ops must be train_op(optimizer related) which will later ignored
  #@TODO allow adding momentum for optimzer
  allow mutliple gpu
  @TODO can we show epoch num info ?
  """
  if optimizer is not None:
    loss = ops[0]
    if isinstance(optimizer, str):
      train_op = melt.gen_train_op_byname(loss, learning_rate, optimizer)
    else:
      train_op = optimizer(learning_rate).minimize(loss) 
    ops = list(ops)
    ops.insert(0, train_op)

  if not model_dir:
    if log_dir and no_log: 
      log_dir = None
    return simple_train_flow(ops, 
                             names, 
                             gen_feed_dict_fn,
                             deal_results_fn,
                             interval_steps,
                             eval_ops,
                             eval_names,
                             gen_eval_feed_dict_fn,
                             deal_eval_results_fn,
                             eval_interval_steps,
                             print_time,
                             print_avg_loss, 
                             log_dir,
                             num_steps,
                             num_steps_per_epoch=num_steps_per_epoch,
                             metric_eval_fn=metric_eval_fn,
                             metric_eval_interval_steps=metric_eval_interval_steps,
                             sess=sess)
  
  #if not set log dir try to use model dir to store log
  #so defaut is write log, if only want save model but disable log, set no_log=True
  if save_model:
    print('Will save model to %s'%model_dir)
  else:
    no_log = True
    print('Will not save model, only read model from %s if exists'%model_dir)
  
  if not log_dir and not no_log:
    log_dir = gezi.get_dir(model_dir)
  if log_dir:
    print('Will save log to %s'%log_dir)
    if add_train_var_histogram:
      # Add histograms for trainable variables. 
      #this is also great for you to see all the trainable variables on tensorboard
      #NOTICE for big model this is too slow!
      melt.monitor_train_vars()
  else:
    print('Will not save log')

  def train_once_(sess, step, is_start=False, fixed_step=None, num_epochs=None, model_path=None):
    train_once(sess, 
               step, 
               ops, 
               names, 
               gen_feed_dict_fn, 
               deal_results_fn, 
               interval_steps, 
               eval_ops, 
               eval_names, 
               gen_eval_feed_dict_fn, 
               deal_eval_results_fn, 
               eval_interval_steps,
               print_time, 
               print_avg_loss,
               model_dir, 
               log_dir, 
               is_start, 
               num_steps_per_epoch,
               metric_eval_fn=metric_eval_fn,
               metric_eval_interval_steps=metric_eval_interval_steps,
               summary_excls=summary_excls,
               fixed_step=fixed_step,
               eval_loops=eval_loops,
               learning_rate=learning_rate,
               learning_rate_patience=learning_rate_patience,
               learning_rate_decay_factor=learning_rate_decay_factor,
               num_epochs=num_epochs,           
               model_path=model_path,
               )
  
  tf_train_flow(train_once_, 
                model_dir, 
                log_dir,
                max_models_keep,
                save_interval_seconds,
                save_interval_steps,
                num_epochs,
                num_steps,
                save_model=save_model,
                save_interval_epochs=save_interval_epochs,
                freeze_graph=freeze_graph,
                num_steps_per_epoch=num_steps_per_epoch,
                restore_from_latest=restore_from_latest,
                metric_eval_fn=metric_eval_fn,
                valid_interval_epochs=valid_interval_epochs,
                inference_fn=inference_fn, 
                inference_interval_epochs=inference_interval_epochs,
                init_fn=init_fn,
                restore_fn=restore_fn,
                restore_include=restore_include,
                restore_exclude=restore_exclude,
                save_all_scope=save_all_scope,
                variables_to_restore=variables_to_restore,
                variables_to_save=variables_to_save,
                output_collection_names=output_collection_names,
                output_node_names=output_node_names,
                learning_rate=learning_rate,
                learning_rate_patience=learning_rate_patience,
                learning_rate_decay_factor=learning_rate_decay_factor,
                write_during_train=write_during_train,
                model=model,
                sess=sess)
