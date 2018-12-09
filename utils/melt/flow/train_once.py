#!/usr/bin/env python
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2016-08-16 12:59:29.331219
#   \Description  
# ==============================================================================

"""
@TODO better logging, using logging.info ?
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six 
if six.PY2:
  from io import BytesIO as IO
else:
  from io import StringIO as IO 

import sys, os, traceback
import inspect

from melt.utils import logging
#import logging

import tensorflow as tf

import gezi
from gezi import Timer, AvgScore 
import melt

projector_config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()

def train_once(sess, 
               step, 
               ops, 
               names=None,
               gen_feed_dict_fn=None, 
               deal_results_fn=None, 
               interval_steps=100,
               eval_ops=None, 
               eval_names=None, 
               gen_eval_feed_dict_fn=None, 
               deal_eval_results_fn=melt.print_results, 
               eval_interval_steps=100, 
               print_time=True, 
               print_avg_loss=True, 
               model_dir=None, 
               log_dir=None, 
               is_start=False,
               num_steps_per_epoch=None,
               metric_eval_fn=None,
               metric_eval_interval_steps=0,
               summary_excls=None,
               fixed_step=None,   # for epoch only, incase you change batch size
               eval_loops=1,
               learning_rate=None,
               learning_rate_patience=None,
               learning_rate_decay_factor=None,
               num_epochs=None,
               model_path=None,
               ):

  #is_start = False # force not to evaluate at first step
  #print('-----------------global_step', sess.run(tf.train.get_or_create_global_step()))
  timer = gezi.Timer()
  if print_time:
    if not hasattr(train_once, 'timer'):
      train_once.timer = Timer()
      train_once.eval_timer = Timer()
      train_once.metric_eval_timer = Timer()
   
  melt.set_global('step', step)
  epoch = (fixed_step or step) / num_steps_per_epoch if num_steps_per_epoch else -1
  if not num_epochs:
    epoch_str = 'epoch:%.3f' % (epoch) if num_steps_per_epoch else ''
  else:
    epoch_str = 'epoch:%.3f/%d' % (epoch, num_epochs) if num_steps_per_epoch else ''
  melt.set_global('epoch', '%.2f' % (epoch))
  
  info = IO()
  stop = False
    
  if eval_names is None:
    if names:
      eval_names = ['eval/' + x for x in names]
  
  if names:
    names = ['train/' + x for x in names]

  if eval_names:
    eval_names = ['eval/' + x for x in eval_names]

  is_eval_step = is_start or eval_interval_steps and step % eval_interval_steps == 0
  summary_str = []
  
  if is_eval_step:
    # deal with summary
    if log_dir:
      if not hasattr(train_once, 'summary_op'):
        #melt.print_summary_ops()
        if summary_excls is None:
          train_once.summary_op = tf.summary.merge_all()
        else:
          summary_ops = []
          for op in tf.get_collection(tf.GraphKeys.SUMMARIES):
            for summary_excl in summary_excls:
              if not summary_excl in op.name:
                summary_ops.append(op)
          print('filtered summary_ops:')
          for op in summary_ops:
            print(op)
          train_once.summary_op = tf.summary.merge(summary_ops)
        
        #train_once.summary_train_op = tf.summary.merge_all(key=melt.MonitorKeys.TRAIN)
        train_once.summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(train_once.summary_writer, projector_config)
    
    if eval_ops is not None:
      #if deal_eval_results_fn is None and eval_names is not None:
      #  deal_eval_results_fn = lambda x: melt.print_results(x, eval_names)
      for i in range(eval_loops):
        eval_feed_dict = {} if gen_eval_feed_dict_fn is None else gen_eval_feed_dict_fn()
        #eval_feed_dict.update(feed_dict)
        
        # might use EVAL_NO_SUMMARY if some old code has problem TODO CHECK
        if not log_dir or train_once.summary_op is None or gezi.env_has('EVAL_NO_SUMMARY'):
        #if not log_dir or train_once.summary_op is None:
          eval_results = sess.run(eval_ops, feed_dict=eval_feed_dict)
        else:
          eval_results = sess.run(eval_ops + [train_once.summary_op], feed_dict=eval_feed_dict)
          summary_str = eval_results[-1]
          eval_results = eval_results[:-1]
        eval_loss = gezi.get_singles(eval_results)
        #timer_.print()
        eval_stop = False

        # @TODO user print should also use logging as a must ?
        #print(gezi.now_time(), epoch_str, 'eval_step: %d'%step, 'eval_metrics:', end='')  
        eval_names_ = melt.adjust_names(eval_loss, eval_names)
        logging.info2('{} eval_step:{} eval_metrics:{}'.format(epoch_str, step, melt.parse_results(eval_loss, eval_names_)))
        
        # if deal_eval_results_fn is not None:
        #   eval_stop = deal_eval_results_fn(eval_results)

        assert len(eval_loss) > 0
        if eval_stop is True:
          stop = True
        eval_names_ = melt.adjust_names(eval_loss, eval_names)
        melt.set_global('eval_loss', melt.parse_results(eval_loss, eval_names_))

    elif interval_steps != eval_interval_steps:
      #print()
      pass

  metric_evaluate = False

  # if metric_eval_fn is not None \
  #   and (is_start \
  #     or (num_steps_per_epoch and step % num_steps_per_epoch == 0) \
  #          or (metric_eval_interval_steps \
  #              and step % metric_eval_interval_steps == 0)):
  #  metric_evaluate = True

  if metric_eval_fn is not None \
    and ((is_start or metric_eval_interval_steps \
         and step % metric_eval_interval_steps == 0) or model_path):
   metric_evaluate = True

  #if (is_start or step == 0) and (not 'EVFIRST' in os.environ):
  if ((step == 0) and (not 'EVFIRST' in os.environ)) or ('QUICK' in os.environ) or ('EVFIRST' in os.environ and os.environ['EVFIRST'] == '0'):
    metric_evaluate = False

  if metric_evaluate:
    # TODO better 
    if not model_path or 'model_path' not in inspect.getargspec(metric_eval_fn).args:
      l = metric_eval_fn()
      if len(l) == 2:
        evaluate_results, evaluate_names = l
        evaluate_summaries = None
      else:
        evaluate_results, evaluate_names, evaluate_summaries = l
    else:
      try:
        l = metric_eval_fn(model_path=model_path)     
        if len(l) == 2:
          evaluate_results, evaluate_names = l
          evaluate_summaries = None
        else:
          evaluate_results, evaluate_names, evaluate_summaries = l
      except Exception:
        logging.info('Do nothing for metric eval fn with exception:\n', traceback.format_exc())
    
    logging.info2('{} valid_step:{} {}:{}'.format(epoch_str, step, 'valid_metrics' if model_path is None else 'epoch_valid_metrics', melt.parse_results(evaluate_results, evaluate_names)))
 
    if learning_rate is not None and (learning_rate_patience and learning_rate_patience > 0):
      assert learning_rate_decay_factor > 0 and learning_rate_decay_factor < 1
      valid_loss = evaluate_results[0]
      if not hasattr(train_once, 'min_valid_loss'):
        train_once.min_valid_loss = valid_loss
        train_once.deacy_steps = []
        train_once.patience = 0
      else:
        if valid_loss < train_once.min_valid_loss:
          train_once.min_valid_loss = valid_loss
          train_once.patience = 0
        else:
          train_once.patience += 1
          logging.info2('{} valid_step:{} patience:{}'.format(epoch_str, step, train_once.patience))
      
      if learning_rate_patience and train_once.patience >= learning_rate_patience:
        lr_op = ops[1]
        lr = sess.run(lr_op) * learning_rate_decay_factor
        train_once.deacy_steps.append(step)
        logging.info2('{} valid_step:{} learning_rate_decay by *{}, learning_rate_decay_steps={}'.format(epoch_str, step, learning_rate_decay_factor, ','.join(map(str, train_once.deacy_steps))))
        sess.run(tf.assign(lr_op, tf.constant(lr, dtype=tf.float32)))
        train_once.patience = 0
        train_once.min_valid_loss = valid_loss

  if ops is not None:
    #if deal_results_fn is None and names is not None:
    #  deal_results_fn = lambda x: melt.print_results(x, names)
    
    feed_dict = {} if gen_feed_dict_fn is None else gen_feed_dict_fn()
    # NOTICE ops[2] should be scalar otherwise wrong!! loss should be scalar
    #print('---------------ops', ops) 
    if eval_ops is not None or not log_dir or not hasattr(train_once, 'summary_op') or train_once.summary_op is None:
      results = sess.run(ops, feed_dict=feed_dict) 
    else:
      #try:
      results = sess.run(ops + [train_once.summary_op], feed_dict=feed_dict)
      summary_str = results[-1]
      results = results[:-1]
      # except Exception:
      #   logging.info('sess.run(ops + [train_once.summary_op], feed_dict=feed_dict) fail')
      #   results = sess.run(ops, feed_dict=feed_dict) 

    #print('------------results', results)
    # #--------trace debug
    # if step == 210:
    #   run_metadata = tf.RunMetadata()
    #   results = sess.run(
    #         ops,
    #         feed_dict=feed_dict,
    #         options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
    #         run_metadata=run_metadata)
    #   from tensorflow.python.client import timeline
    #   trace = timeline.Timeline(step_stats=run_metadata.step_stats)

    #   trace_file = open('timeline.ctf.json', 'w')
    #   trace_file.write(trace.generate_chrome_trace_format())
    
    #reults[0] assume to be train_op, results[1] to be learning_rate
    learning_rate = results[1]
    results = results[2:]

    #@TODO should support aver loss and other avg evaluations like test..
    if print_avg_loss:
      if not hasattr(train_once, 'avg_loss'):
        train_once.avg_loss = AvgScore()
        if interval_steps != eval_interval_steps:
          train_once.avg_loss2 = AvgScore()
      #assume results[0] as train_op return, results[1] as loss
      loss = gezi.get_singles(results)
      train_once.avg_loss.add(loss)
      if interval_steps != eval_interval_steps:
        train_once.avg_loss2.add(loss)
    
    steps_per_second = None
    instances_per_second = None 
    hours_per_epoch = None
    #step += 1
    if is_start or interval_steps and step % interval_steps == 0:
      train_average_loss = train_once.avg_loss.avg_score()
      if print_time:
        duration = timer.elapsed()
        duration_str = 'duration:{:.3f} '.format(duration)
        melt.set_global('duration', '%.3f' % duration)
        info.write(duration_str)
        elapsed = train_once.timer.elapsed()
        steps_per_second = interval_steps / elapsed
        batch_size = melt.batch_size()
        num_gpus = melt.num_gpus()
        instances_per_second = interval_steps * batch_size / elapsed
        gpu_info = '' if num_gpus <= 1 else ' gpus:[{}]'.format(num_gpus)
        if num_steps_per_epoch is None:
          epoch_time_info = ''
        else:
          hours_per_epoch = num_steps_per_epoch / interval_steps * elapsed / 3600
          epoch_time_info = ' 1epoch:[{:.2f}h]'.format(hours_per_epoch)
        info.write('elapsed:[{:.3f}] batch_size:[{}]{} batches/s:[{:.2f}] insts/s:[{:.2f}] {} lr:[{:.8f}]'.format(
                      elapsed, batch_size, gpu_info, steps_per_second, instances_per_second, epoch_time_info, learning_rate))

      if print_avg_loss:
        #info.write('train_avg_metrics:{} '.format(melt.value_name_list_str(train_average_loss, names)))
        names_ = melt.adjust_names(train_average_loss, names)
        #info.write('train_avg_metric:{} '.format(melt.parse_results(train_average_loss, names_)))
        info.write(' train:{} '.format(melt.parse_results(train_average_loss, names_)))
        #info.write('train_avg_loss: {} '.format(train_average_loss))
      
      #print(gezi.now_time(), epoch_str, 'train_step:%d'%step, info.getvalue(), end=' ') 
      logging.info2('{} {} {}'.format(epoch_str, 'step:%d'%step, info.getvalue()))
      
      if deal_results_fn is not None:
        stop = deal_results_fn(results)

  summary_strs = gezi.to_list(summary_str)  
  if metric_evaluate:
    if evaluate_summaries is not None:
      summary_strs += evaluate_summaries 

  if step > 1:
    if is_eval_step:
      # deal with summary
      if log_dir:
        # if not hasattr(train_once, 'summary_op'):
        #   melt.print_summary_ops()
        #   if summary_excls is None:
        #     train_once.summary_op = tf.summary.merge_all()
        #   else:
        #     summary_ops = []
        #     for op in tf.get_collection(tf.GraphKeys.SUMMARIES):
        #       for summary_excl in summary_excls:
        #         if not summary_excl in op.name:
        #           summary_ops.append(op)
        #     print('filtered summary_ops:')
        #     for op in summary_ops:
        #       print(op)
        #     train_once.summary_op = tf.summary.merge(summary_ops)

        #   print('-------------summary_op', train_once.summary_op)
          

        #   #train_once.summary_train_op = tf.summary.merge_all(key=melt.MonitorKeys.TRAIN)
        #   train_once.summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        #   tf.contrib.tensorboard.plugins.projector.visualize_embeddings(train_once.summary_writer, projector_config)

        summary = tf.Summary()
        # #so the strategy is on eval_interval_steps, if has eval dataset, then tensorboard evluate on eval dataset
        # #if not have eval dataset, will evaluate on trainset, but if has eval dataset we will also monitor train loss
        # assert train_once.summary_train_op is None
        # if train_once.summary_train_op is not None:
        #   summary_str = sess.run(train_once.summary_train_op, feed_dict=feed_dict)
        #   train_once.summary_writer.add_summary(summary_str, step)

        if eval_ops is None:
          # #get train loss, for every batch train
          # if train_once.summary_op is not None:
          #   #timer2 = gezi.Timer('sess run')
          #   try:
          #     # TODO FIXME so this means one more train batch step without adding to global step counter ?! so should move it earlier 
          #     summary_str = sess.run(train_once.summary_op, feed_dict=feed_dict)
          #   except Exception:
          #     if not hasattr(train_once, 'num_summary_errors'):
          #       logging.warning('summary_str = sess.run(train_once.summary_op, feed_dict=feed_dict) fail')
          #       train_once.num_summary_errors = 1
          #       logging.warning(traceback.format_exc())
          #     summary_str = ''
          #   # #timer2.print()
          if train_once.summary_op is not None:
            for summary_str in summary_strs:
              train_once.summary_writer.add_summary(summary_str, step)
        else:
          # #get eval loss for every batch eval, then add train loss for eval step average loss
          # try:
          #   summary_str = sess.run(train_once.summary_op, feed_dict=eval_feed_dict) if train_once.summary_op is not None else ''
          # except Exception:
          #   if not hasattr(train_once, 'num_summary_errors'):
          #     logging.warning('summary_str = sess.run(train_once.summary_op, feed_dict=eval_feed_dict) fail')
          #     train_once.num_summary_errors = 1
          #     logging.warning(traceback.format_exc())
          #   summary_str = ''
          #all single value results will be add to summary here not using tf.scalar_summary..
          #summary.ParseFromString(summary_str)
          for summary_str in summary_strs:
            train_once.summary_writer.add_summary(summary_str, step)
          suffix = 'eval' if not eval_names else ''
          melt.add_summarys(summary, eval_results, eval_names_, suffix=suffix)

        if ops is not None:
          melt.add_summarys(summary, train_average_loss, names_, suffix='train_avg') 
          ##optimizer has done this also
          melt.add_summary(summary, learning_rate, 'learning_rate')
          melt.add_summary(summary, melt.batch_size(), 'batch_size')
          melt.add_summary(summary, melt.epoch(), 'epoch')
          if steps_per_second:
            melt.add_summary(summary, steps_per_second, 'steps_per_second')
          if instances_per_second:
            melt.add_summary(summary, instances_per_second, 'instances_per_second') 
          if hours_per_epoch:
            melt.add_summary(summary, hours_per_epoch, 'hours_per_epoch')

        if metric_evaluate:
          #melt.add_summarys(summary, evaluate_results, evaluate_names, prefix='eval')
          prefix = 'step/valid'
          if model_path:
            prefix = 'epoch/valid'
            if not hasattr(train_once, 'epoch_step'):
              train_once.epoch_step = 1
            else:
              train_once.epoch_step += 1
            step = train_once.epoch_step
            
          melt.add_summarys(summary, evaluate_results, evaluate_names, prefix=prefix)
        
        train_once.summary_writer.add_summary(summary, step)
        train_once.summary_writer.flush()

        #timer_.print()
      
      # if print_time:
      #   full_duration = train_once.eval_timer.elapsed()
      #   if metric_evaluate:
      #     metric_full_duration = train_once.metric_eval_timer.elapsed()
      #   full_duration_str = 'elapsed:{:.3f} '.format(full_duration)
      #   #info.write('duration:{:.3f} '.format(timer.elapsed()))
      #   duration = timer.elapsed()
      #   info.write('duration:{:.3f} '.format(duration))
      #   info.write(full_duration_str)
      #   info.write('eval_time_ratio:{:.3f} '.format(duration/full_duration))
      #   if metric_evaluate:
      #     info.write('metric_time_ratio:{:.3f} '.format(duration/metric_full_duration))
      # #print(gezi.now_time(), epoch_str, 'eval_step: %d'%step, info.getvalue())
      # logging.info2('{} {} {}'.format(epoch_str, 'eval_step: %d'%step, info.getvalue()))
      return stop
    elif metric_evaluate:
      summary = tf.Summary()
      for summary_str in summary_strs:
        train_once.summary_writer.add_summary(summary_str, step)
      #summary.ParseFromString(evaluate_summaries)
      summary_writer = train_once.summary_writer
      prefix = 'step/valid'
      if model_path:
        prefix = 'epoch/valid'
        if not hasattr(train_once, 'epoch_step'):
          ## TODO.. restart will get 1 again..
          #epoch_step = tf.Variable(0, trainable=False, name='epoch_step')
          #epoch_step += 1
          #train_once.epoch_step = sess.run(epoch_step) 
          valid_interval_epochs = 1. 
          try:
            valid_interval_epochs = FLAGS.valid_interval_epochs 
          except Exception:
            pass
          train_once.epoch_step = 1 if melt.epoch() <= 1 else int(int(melt.epoch() * 10) / int(valid_interval_epochs * 10))
          logging.info('train_once epoch start step is', train_once.epoch_step)
        else:
          #epoch_step += 1
          train_once.epoch_step += 1
        step = train_once.epoch_step
      #melt.add_summarys(summary, evaluate_results, evaluate_names, prefix='eval')
      melt.add_summarys(summary, evaluate_results, evaluate_names, prefix=prefix)
      summary_writer.add_summary(summary, step)
      summary_writer.flush()
