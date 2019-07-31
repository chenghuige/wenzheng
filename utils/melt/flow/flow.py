  #!/usr/bin/env python
# ==============================================================================
#          \file   flow.py
#        \author   chenghuige  
#          \date   2016-08-17 10:48:46.141744
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
#import gflags as flags
FLAGS = flags.FLAGS

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import tpu
try:
  import horovod.tensorflow as hvd
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
except Exception:
  pass

import os, sys, traceback
import melt 
import gezi
from melt.utils import logging
import glob
import inspect
import traceback
import numpy as np

def tf_flow(process_once, model_dir=None, num_steps=None, sess=None):
  """
  basic flow for tf records, allow most freedom for usage, if not tfrecords no need for flow
  Args:
  train_once: function with 2 inputs sess and step
  """
  init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
  if sess is None:
    sess = tf.InteractiveSession()

  if not model_dir:
    sess.run(init_op)
  else:
    melt.restore(sess, model_dir)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  # try:
  #   # dataset
  #   print('tf_flow using datset')
  #   step = 0
  #   try:
  #     while True:
  #       stop = process_once(sess, step)
  #       if stop is True:
  #         print('Early stop running %d stpes'%(step))
  #         raise tf.errors.OutOfRangeError(None, None,'Early stop running %d stpes'%(step))
  #       step += 1
  #       if num_steps and step == num_steps:
  #         raise tf.errors.OutOfRangeError(None, None, 'Reached max num steps')
  #   except tf.errors.OutOfRangeError:
  #     print('Done training for %d steps.' % (step))
  # except Exception:
  #   # old queue method
  #   print('tf_flow using queue')
  try:
    step = 0
    while not coord.should_stop():
      stop = process_once(sess, step)
      if stop is True:
        print('Early stop running %d stpes'%(step))
        raise tf.errors.OutOfRangeError(None, None,'Early stop running %d stpes'%(step))
      step += 1
      if num_steps and step == num_steps:
        raise tf.errors.OutOfRangeError(None, None, 'Reached max num steps')
  except tf.errors.OutOfRangeError:
    print('Done training for %d steps.' % (step))
  finally:
    coord.request_stop()
  coord.join(threads)

  sess.close()
  return step
 
def _get_model_path(model_dir, save_model):
  if os.path.exists(model_dir + '.index') and os.path.exists(model_dir + '.meta'):
    # input is real model path
    return model_dir
  if not os.path.exists(model_dir):
    if save_model:
      gezi.try_mkdir(model_dir)
    return None
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    #input valid dir and return latest model
    return os.path.join(model_dir, os.path.basename(ckpt.model_checkpoint_path))
  elif os.path.isdir(model_dir):
    #input valid dir but no models
    return None 
  else:
    #this might be user specified model like ./model/model-100.ckpt
    #the file exists and we NOTICE we do not check if it is valid model file!
    return model_dir

def _get_checkpoint_path(checkpoint_path, step=None, num_steps_per_epoch=None, epoch=None):
  if not num_steps_per_epoch:
    return checkpoint_path
  return '%s-%.2f'%(checkpoint_path, epoch or step / float(num_steps_per_epoch))

def tf_train_flow(train_once_fn, 
                  model_dir=None,
                  log_dir=None, 
                  max_models_keep=1, 
                  save_interval_seconds=600, 
                  save_interval_steps=1000, 
                  num_epochs=None,
                  num_steps=None, 
                  save_model=True,
                  save_interval_epochs=None, 
                  freeze_graph=False,
                  num_steps_per_epoch=0,
                  restore_from_latest=True,
                  metric_eval_fn=None,
                  valid_interval_epochs=0,
                  inference_fn=None, 
                  inference_interval_epochs=0,
                  init_fn=None,
                  restore_fn=None,
                  restore_include=None,
                  restore_exclude=None,
                  save_all_scope=False, #TODO save load from restore scope only but svae all
                  variables_to_restore=None,
                  variables_to_save=None, #by default will be the same as variables_to_restore
                  output_collection_names=None, 
                  output_node_names=None,
                  learning_rate=None, #not use yet, just use in train_once
                  learning_rate_patience=None,
                  learning_rate_decay_factor=None,
                  write_during_train=True,
                  model=None,
                  sess=None):
  """
  similary flow as tf_flow, but add model try reload and save
  """
  use_horovod = 'OMPI_COMM_WORLD_RANK' in os.environ

  model_dir_ = model_dir
  if use_horovod and hvd.rank != 0:
    model_dir = None
  
  if sess is None:
    #TODO melt.get_session is global session but may cause non close at last
    sess = melt.get_session()

  if FLAGS.use_tpu:
    sess.run(tpu.initialize_system())
  #logging.info('tf_train_flow start')
  #logging.info('max_models_keep:', max_models_keep)
  #logging.info('save_interval_seconds:', save_interval_seconds)

  if model_dir:
    if model:
      checkpoint = tf.train.Checkpoint(model=model)
      ckpt_dir = model_dir + '/ckpt'
      checkpoint_prefix = os.path.join(ckpt_dir, 'ckpt')
  
    #this is usefull for you use another model with another scope, and just load and restore/save initalize your scope vars!
    #this is not for finetune but mainly for like using another model as in predict like this introducing graph other model scope and ignore here

    # var_list = None if not restore_scope else tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=restore_scope)
    # #logging.info('-------------var_list', var_list)
    
    # if not variables_to_restore:
    #   variables_to_restore = var_list

    if not variables_to_restore:
      variables_to_restore = slim.get_variables_to_restore(include=restore_include, exclude=restore_exclude)

    if not variables_to_save:
      variables_to_save = variables_to_restore
    if save_all_scope:
      variables_to_save = None
    
    #if variables_to_restore is None:
    logging.info('variables_to_restore from %s' % model_dir)
    #load all var in checkpoint try to save all var(might more then original checkpoint) if not specifiy variables_to_save
    varnames_in_checkpoint = melt.get_checkpoint_varnames(model_dir)
    #logging.info('varnames_in_checkpoint: {}'.format(varnames_in_checkpoint))

    # TODO has someproblem say  tf.Variable 'r_net/text_encoder/cudnn_rnn/cu_dnngru/recurrent_kernel/adam_v:0' even though in checkpoint I have renated it as ignore/rnet
    variables_to_restore_from_model = slim.get_variables_to_restore(include=varnames_in_checkpoint)
    #logging.info('variables_to_restore_from_model: {}'.format(variables_to_restore_from_model))
    if not variables_to_restore:
      variables_to_restore = variables_to_restore_from_model
    else:
      variables_to_restore = [v for v in variables_to_restore if v in variables_to_restore_from_model]
    if restore_exclude:
      for excl in restore_exclude:
        variables_to_restore = [v for v in  variables_to_restore if not excl in v.name]
    #--tf 1.6 adadelta will have same vars... 
    variables_to_restore = list(set(variables_to_restore))
    #logging.info('variables_to_restore', variables_to_restore[:100])
    logging.info('variables_to_restore', [x for x in variables_to_restore if not 'OptimizeLoss' in x.name][:100])

  ##finally remove global_step since melt.apps.train will handle it!
  global_step = tf.train.get_or_create_global_step()
  
  #variables_to_restore = [v for v in variables_to_restore if not tf.GraphKeys.GLOBAL_STEP in v.name]
  #variables_to_restore = [v for v in variables_to_restore if not 'learning_rate' in v.name]

  # TODO fixme if step, step2.. and in checkpoint step then here will be step2...
  #print('------------', [v for v in variables_to_restore if 'step' in v.name])
  loader = tf.train.Saver(var_list=variables_to_restore) 

  logging.info('max models to keep {}, keep every {} hours'.format(max_models_keep, save_interval_seconds / 3600.0))
  saver = tf.train.Saver(
    max_to_keep=max_models_keep, 
    keep_checkpoint_every_n_hours=save_interval_seconds / 3600.0,
    var_list=variables_to_save) 
  epoch_saver = tf.train.Saver(var_list=variables_to_save, max_to_keep=1000)
  best_epoch_saver = tf.train.Saver(var_list=variables_to_save) 
  #logging.info('variables_to_save:{}'.format(variables_to_save))

  # # #TODO for safe restore all init will be ok ?
  # # if variables_to_restore is None:
  init_op = tf.group(tf.global_variables_initializer(), #variables_initializer(global_variables())
                     tf.local_variables_initializer()) #variables_initializer(local_variables())
  # # else:
  # #   init_op = tf.group(tf.variables_initializer(variables_to_restore),
  # #                      tf.local_variables_initializer())
  
  ##--mostly this will be fine except for using assistant predictor, initialize again! will make assistant predictor wrong
  ##so assume to all run init op! if using assistant predictor, make sure it use another session
  
  # https://stackoverflow.com/questions/35164529/in-tensorflow-is-there-any-way-to-just-initialize-uninitialised-variables
  # def guarantee_initialized_variables(session, list_of_variables = None):
  #   if list_of_variables is None:
  #       list_of_variables = tf.global_variables()
  #   uninitialized_variables = list(tf.get_variable(name) for name in
  #                                  session.run(tf.report_uninitialized_variables(list_of_variables)))
  #   return unintialized_variables

  # unintialized_variables = guarantee_initialized_variables(sess)
  # init_op = tf.group(tf.initialize_variables(uninitialized_vars), tf.local_variables_initializer())

  timer = gezi.Timer('sess run init_op in melt.tf_train_flow')
  #model.save('./weights')

  # notice 
  sess.run(init_op)

  timer.print_elapsed()

  #melt.init_uninitialized_variables(sess)

  #pre_step means the step last saved, train without pretrained,then -1
  pre_step = -1
  fixed_pre_step = -1  #fixed pre step is for epoch num to be correct if you change batch size
  #print(model_dir)
  pre_epoch = None
  if model_dir:
    model_path = _get_model_path(model_dir, save_model)
    #print(model_path)
    model_dir = gezi.get_dir(model_dir) #incase you pass ./model/model-ckpt1000 -> ./model
  
    if model_path is not None:
      if not restore_from_latest:
        logging.info('using recent but not latest model')
        model_path = melt.recent_checkpoint(model_dir)
      model_name = os.path.basename(model_path)
      timer = gezi.Timer('Loading and training from existing model [%s]' % model_path)
      if restore_fn is not None:
        restore_fn(sess)
      loader.restore(sess, model_path)
      ## not supported
      #model.save()
      #model.save_weights('./weights')
      timer.print()
      #pre_step = melt.get_model_step(model_path) - 1 if FLAGS.global_step is None else FLAGS.global_step -1
      # TODO check ..
      pre_step = sess.run(tf.train.get_global_step()) - 1
      pre_epoch = melt.get_model_epoch(model_path) if FLAGS.global_epoch is None else FLAGS.global_epoch
      fixed_pre_step = pre_step
      # if pre_epoch is not None:
      #   #like using batch size 32, then reload train using batch size 64
      #   if abs(pre_step / num_steps_per_epoch - pre_epoch) > 0.1:
      #     fixed_pre_step = int(pre_epoch * num_steps_per_epoch)
      #     logging.info('Warning, epoch is diff with pre_step / num_steps_per_epoch:{}, pre_epoch:{},maybe you change batch size and we will adjust to set pre_step as {}'\
      #       .format(pre_step / num_steps_per_epoch, pre_epoch, fixed_pre_step))
    else:
      latest_checkpoint = None
      try:
        latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        if latest_checkpoint:
          logging.info('Try start from eager trained mode, latest checkpoint:', latest_checkpoint)
          checkpoint.restore(latest_checkpoint).run_restore_ops(session=sess)

          pre_epoch = int(latest_checkpoint.split('-')[-1])
          #pre_step = pre_epoch * num_steps_per_epoch - 1
          # TODO check
          pre_step = sess.run(tf.train.get_global_step()) - 1
          fixed_pre_step = pre_step
          logging.info('Start step is:', pre_step)
      except Exception:
        logging.info('Something wrong with restore from eager trained model')
      if latest_checkpoint is None:
        logging.info('Train all start step 0')
        #https://stackoverflow.com/questions/40220201/tensorflow-tf-initialize-all-variables-vs-tf-initialize-local-variables
        #tf.initialize_all_variables() is a shortcut to tf.initialize_variables(tf.all_variables()), 
        #tf.initialize_local_variables() is a shortcut to tf.initialize_variables(tf.local_variables()), 
        #which initializes variables in GraphKeys.VARIABLES and GraphKeys.LOCAL_VARIABLE collections, respectively.
        #init_op = tf.group(tf.global_variables_initializer(),
        #                   tf.local_variables_initializer())   
        #[var for var in tf.all_variables() if var.op.name.startswith(restore_scope)] will be the same as tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=restore_scope)
        
        #sess.run(init_op)

        #like use image model, build image graph, reload first train, and then will go to same checkpoint all varaible just restore will ok
        #for finetune from loading other model init
        if init_fn is not None:
          init_fn(sess)
        
  if gezi.env_has('METRIC'):
    l = metric_eval_fn(model_path)
    print(list(zip(l[1], l[0])))
    exit(0)

  #sess.run(tf.assign(global_step, tf.constant(global_step_val, dtype=tf.int64)))
  try:
    learning_rate = tf.get_collection('learning_rate')[-1]
    learning_rate_weight = tf.get_collection('learning_rate_weight')[-1]
    sess.run(tf.assign(learning_rate, learning_rate * learning_rate_weight))
  except Exception:
    # if not using weight_decay but using optimizer decay then will go here as learning rate is a tensor can not assign
    pass

  try:
    logging.info('Actual start global step:', sess.run(global_step), 'learning rate:', sess.run(learning_rate), 'learning_rate_weight:', sess.run(learning_rate_weight))
  except Exception:
    pass
  
  if model_dir_:
    #if save_interval_epochs and num_steps_per_epoch and num_steps >= 0:
    epoch_dir = os.path.join(model_dir_, 'epoch')
    gezi.try_mkdir(epoch_dir)
    checkpoint_path = os.path.join(model_dir_, 'model.ckpt')
  
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  if use_horovod:
    bcast = hvd.broadcast_global_variables(0)
    sess.run(bcast)

  #tf.train.write_graph(sess.graph_def, model_dir, 'train.pbtxt')
  only_one_step = False
  try:
    if use_horovod:
      ## TODO FIXME why bcast here not work ? simple test work see tests/bcast.py
      #comm.bcast(pre_step, root=0)
      temp = np.array([pre_step])
      comm.Bcast(temp, root=0)
      pre_step = temp[0]

    step = start = pre_step + 1
    fixed_step = fixed_pre_step + 1 

    #first = True

    #hack just for save one model after load
    if num_steps < 0 or (num_steps and num_steps < step):
      logging.info('just load and resave then exit')
      model_path_ =  _get_checkpoint_path(checkpoint_path, step, num_steps_per_epoch, epoch=pre_epoch)
      saver.save(sess, model_path_, global_step=step + 1)
      if freeze_graph:
        melt.freeze_graph(sess, model_path_, step + 1, output_collection_names, output_node_names)
      sess.close()
      exit(0)
    
    if num_epochs < 0:
      only_one_step = True
      logging.info('just run one step')

    if FLAGS.work_mode != 'train':
      assert not os.path.isdir(FLAGS.model_dir), FLAGS.model_dir  
      if 'valid' in FLAGS.work_mode:
        vals, names = metric_eval_fn(FLAGS.model_dir)
        logging.info(list(zip(names, vals)))
      if 'test' in FLAGS.work_mode:
        inference_fn(FLAGS.model_dir)
      exit(0)

    #early_stop = True #TODO allow config
    num_bad_epochs = 0
    pre_epoch_eval_loss = 1e20
    best_epoch_eval_loss = 1e20
    num_allowed_bad_epochs = 4 #allow 5 non decrease eval loss epochs  before stop
    epoch_saved_step = 0
    while not coord.should_stop():
      model_step_path = None
      if model_dir_:
        model_path_ = os.path.join(epoch_dir,'model.ckpt-%.2f'%(fixed_step / float(num_steps_per_epoch)))
        model_step_path_ = model_path_ + '-' + str(step)
        if (write_during_train and metric_eval_fn is not None and valid_interval_epochs and fixed_step % int(num_steps_per_epoch * valid_interval_epochs) == 0):
          model_step_path = model_step_path_
        else:
           model_step_path = None

      if step == 0:
        model_step_path = None

      #print('--------------------step', step)
      stop = train_once_fn(sess, 
                           step, 
                           is_start=(step==start), 
                           fixed_step=fixed_step,
                           num_epochs=num_epochs,
                           model_path=model_step_path,
                           use_horovod=use_horovod,
                           ## TODO FIXME this line will cause   tensorflow.python.framework.errors_impl.NotFoundError: Resource localhost/save_counter/N10tensorflow3VarE does not exist. 
                          )


      #first = False

      if only_one_step:
        stop = True

      step += 1
      fixed_step += 1
      if save_model and step and model_dir:
        #step 0 is also saved! actually train one step and save
        if step % save_interval_steps == 0:
          timer = gezi.Timer('save model step %d to %s'%(step, checkpoint_path), False)
          model_path_ = _get_checkpoint_path(checkpoint_path, fixed_step, num_steps_per_epoch)
          saver.save(sess, model_path_, global_step=step)
          if freeze_graph:
            melt.freeze_graph(sess, model_path_, step, output_collection_names, output_node_names)
          #if log_dir != model_dir:
          #  assert log_dir
          #  command = 'rsync -l -r -t %s/* %s' % (log_dir, model_dir) 
          #  print(command, file=sys.stderr)
          #  os.system(command)
          timer.print_elapsed()
  
        if save_interval_steps and num_steps_per_epoch and fixed_step % int(num_steps_per_epoch * save_interval_epochs) == 0:
          # TODO only epoch in name not sep ?
          epoch_saved_step = step
          model_path_ = os.path.join(epoch_dir,'model.ckpt-%.2f'%(fixed_step / float(num_steps_per_epoch)))
          model_step_path = model_path_ + '-' + str(step)
          epoch_saver.save(sess, model_path_, global_step=step)
          #epoch_saver.save(sess, model_path_)
          
          if model:
            #model.save_weights(epoch_dir + '/ckpt-%.2f' % (fixed_step / float(num_steps_per_epoch)))
            # TODO FIXME if restart will save from 1... again..
            checkpoint.save(checkpoint_prefix, session=sess)
            #print(sess.run(checkpoint.save_counter))
            
          if freeze_graph:
            melt.freeze_graph(sess, model_path_, step, output_collection_names, output_node_names)

        if write_during_train:
          if inference_fn is not None and inference_interval_epochs and fixed_step % int(num_steps_per_epoch * inference_interval_epochs) == 0:
            model_step_path = model_path_ + '-' + str(step)
            try:
              #print('--------------inference fn')
              inference_fn(model_path=model_step_path)
            except Exception:
              logging.info(traceback.format_exc())  

          # if metric_eval_fn is not None and valid_interval_epochs and fixed_step % int(num_steps_per_epoch * valid_interval_epochs) == 0:
          #   model_step_path = model_path_ + '-' + str(step)
          #   try:
          #     metric_eval_fn(model_path=model_step_path)
          #   except Exception:
          #     logging.info(traceback.format_exc())  

      if stop is True:
        print('Early stop running %d stpes'%(step), file=sys.stderr)
        raise tf.errors.OutOfRangeError(None, None,'Early stop running %d stpes'%(step))
      if num_steps and (step + 1) == start + num_steps:
        raise tf.errors.OutOfRangeError(None, None,'Reached max num steps')
      #max_num_epochs = 1000
      max_num_epochs = num_epochs
      #if max_num_epochs and num_steps_per_epoch and fixed_step // num_steps_per_epoch >= max_num_epochs:
      if max_num_epochs and num_steps_per_epoch and fixed_step / num_steps_per_epoch > max_num_epochs:
        raise tf.errors.OutOfRangeError(None, None,'Reached max num epochs of %d'%max_num_epochs)
  #except tf.errors.OutOfRangeError, e:
  except tf.errors.OutOfRangeError:
    # if run 2 epoch and we have just epoch saved, do not need to save only 1 step more model
    if (step - epoch_saved_step > 1) and not (step==start) and save_model and step % save_interval_steps != 0 and model_dir:
      model_path_ = _get_checkpoint_path(checkpoint_path, step, num_steps_per_epoch)
      saver.save(sess, model_path_, global_step=step)
      if freeze_graph:
        melt.freeze_graph(sess, model_path_, step, output_collection_names, output_node_names)
      if log_dir != model_dir:
        assert log_dir
        command = 'rsync -l -r -t %s/* %s' % (log_dir, model_dir) 
        print(command, file=sys.stderr)
        os.system(command)
    if only_one_step:
      logging.info('Done one step')
      exit(0)
    
    if (step - epoch_saved_step > 1) and metric_eval_fn is not None:
      metric_eval_fn(model_path=model_step_path)
    
    if (num_epochs and fixed_step / num_steps_per_epoch >= num_epochs) or (num_steps and step == start + num_steps) :
      logging.info('Done training for %.3f epochs, %d steps.' % (fixed_step / num_steps_per_epoch, step))
      #FIXME becase coord.join seems not work,  RuntimeError: Coordinator stopped with threads still running: Thread-9
      exit(0)
    else:
      logging.info('Should not stop, but stopped at epoch: %.3f'%(fixed_step / num_steps_per_epoch))
      logging.info(traceback.format_exc())
      #raise e
  finally:
    coord.request_stop()

  coord.join(threads, stop_grace_period_secs=5)
  #FIMXE due to use melt.get_session(global not handle del well)
  #Done training for 3090020 steps.
  #Exception TypeError: "'NoneType' object is not callable" in <bound method Session.__del__ of <tensorflow.python.client.session.Session object at 0x7f6cf33cd450>> ignored
  if FLAGS.use_tpu:
    sess.run(tpu.shutdown_system())
  sess.close()

#@TODO not tested yet
def tf_test_flow(test_once, model_dir='./model', 
                 model_name=None, num_epochs=1, num_steps=0,
                 sess=None):
  """
  basic flow for tf records, allow most freedom for usage, if not tfrecords no need for flow
  Args:
  test_once: function with 2 inputs sess and step
  model_dir: can be dir like ./model will fetch lates model in model dir , or be real model path like ./model/model.0.ckpt
  """
  if sess is None:
    sess = tf.InteractiveSession()

  melt.restore(sess, model_dir, model_name)

  if not os.path.isdir(model_dir):
    model_dir = os.path.dirname(model_dir)
  summary_op = tf.merge_all_summaries()
  summary_writer = tf.train.SummaryWriter(model_dir, sess.graph)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  try:
    step = 0
    while not coord.should_stop():
      test_once(sess, step)
      step += 1
      if num_steps and step == num_steps:
        raise tf.errors.OutOfRangeError(None, None, 'Reached max num steps')
  except tf.errors.OutOfRangeError:
    print('Done testing for %d epochs, %d steps.' % (num_epochs, step))
  finally:
    # When done, ask the threads to stop.
    coord.request_stop()
  # Wait for threads to finish.
  coord.join(threads)
