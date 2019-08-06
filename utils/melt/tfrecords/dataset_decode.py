#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   read_sparse.py
#        \author   chenghuige  
#          \date   2016-08-15 20:13:06.751843
#   \Description  @TODO https://github.com/tensorflow/tensorflow/tree/r0.10/tensorflow/contrib/slim/python/slim/data/
# ==============================================================================
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

import gezi
import melt
logging = melt.logging

import sys
import os
import numpy as np
import inspect

# TODO
# #https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/data_reader.py
def padded_batch(d, batch_size, padded_shapes=None):
  padded_shapes = padded_shapes or dict(
      [(name, [None] * len(shape))
       for name, shape in d.output_shapes.items()])
  return d.padded_batch(batch_size, padded_shapes)

def inputs(files, 
           decode_fn, 
           batch_size=64,
           num_epochs = None, 
           num_threads=None, 
           buffer_size = 15000, #change from 1000 to 15000
           dynamic_pad=True,
           shuffle_files=True, batch_join=True, shuffle_batch=True, 
           min_after_dequeue=None, seed=None, enqueue_many=False,
           fix_random=False, no_random=False, fix_sequence=False,
           allow_smaller_final_batch=True, 
           num_prefetch_batches=None, 
           bucket_boundaries=None,
           length_index=None,
           length_key=None,
           length_fn=None,
           bucket_batch_sizes=None,
           repeat=True,
           initializable=False,
           filter_fn=None,
           balance_pos_neg=False,
           pos_filter_fn=None,
           neg_filter_fn=None,
           count_fn=None,
           return_iterator=False,
           Dataset=None,
           batch_parse=False, #by default will be line parse
           hvd_shard=True,
           shard_by_files=False,
           training=True,
           simple_parse=False,
           repeat_then_shuffle=False,
           name='input'):
  """Reads input data num_epochs times.
  for sparse input here will do:
  1. read serialized_example
  2. shuffle serialized_examples
  3. decdoe batch_serialized_examples
  notice read_sparse.inputs and also be used for dense inputs,but if you 
  only need to decode part from serialized_example, then read.inputs will 
  be better, less to put to suffle
  #--------decode example, can refer to libsvm-decode.py
  # def decode(batch_serialized_examples):
  #   features = tf.parse_example(
  #       batch_serialized_examples,
  #       features={
  #           'label' : tf.FixedLenFeature([], tf.int64),
  #           'index' : tf.VarLenFeature(tf.int64),
  #           'value' : tf.VarLenFeature(tf.float32),
  #       })

  #   label = features['label']
  #   index = features['index']
  #   value = features['value']

  #   return label, index, value 

  #string_input_reducer will shuffle files
  #shuffle will read file by file and shuffle withn file(in shuffle queue) 
  #shuffle_batch_join will read multiple files and shuffle in shuffle queue(from many files)

  To get fixed sequence 
  shuffle=False  so by this way the sequence is as your data input unchange
  or
  shuffle=True
  seed=1024 #set
  batch_join=False  by this way you have fixed random, so get same result
  NOTICE, shuffle=True,seed=1024,batch_join=True will not get same result
  shuffle=False,seed=1024,batch_join=True also, so batch_join seems seed only control inqueue random, can not get fixed result

  for no random -> fixed result set shuffle=False wihch will force batch_join=False then use batch
  for fixed random ->  shuffle=True, seed set or  fix_random=True
  read-records.py show above ok, but train-evaluate.py show not, only shuffle=False can get fixed result.. @FIXME strange
  for train-evaluate.py it looks you can set shuffle in string_input_producer True, but then must use batch,
  batch_join and shuffle_batch join all not fixed even with seed set, may be due to trainset two inputs read ?
  for read-records.py batch_join will be fixed, shuffle_batch_join not 

  defualt parmas will give max random...

  Args:
  decode: user defined decode 
  min_after_dequeue: set to >2w for production train, suggesed will be 0.4 * num_instances, but also NOTICE do not exceed mem
  #--default parmas will make most randomness
  shuffle_files: wehter shuffle file 
  shuffle_batch: batch or shuffle_batch
  batch_join: wether to use multiple reader or use one reader mutlitple thread
  fix_random: if True make at most random which can fix random result
  allow_smaller_final_batch: set True usefull if you want verify on small d

  great article http://d0evi1.com/tensorflow/ds_performance/
  https://www.tensorflow.org/versions/master/performance/ds_performance
  """
  use_horovod = 'OMPI_COMM_WORLD_RANK' in os.environ
  if use_horovod:
    if FLAGS.torch:
      import horovod.torch as hvd
    else:
      import horovod.tensorflow as hvd

  def shard(d):
    return d.shard(hvd.size(), hvd.rank())

  # Choose to use cpu outside input function like in d.py
  #with tf.device('/cpu:0'):
  if isinstance(files, str):
    files = gezi.list_files(files)
  assert len(files) > 0

  if use_horovod and not hvd_shard and training:
    assert len(files) % hvd.size() == 0, '{} {} {}'.format(len(files), files, hvd.size())
    files_ = []
    for i in range(len(files)):
      if i % hvd.size() == hvd.rank():
        files_.append(files[i])
    files = files_
    print('----------train-files', files)
    #exit(0)

  if not num_threads:
    try:
      import multiprocessing
      #num_threads = int(multiprocessing.cpu_count() * 0.6)
      num_threads = multiprocessing.cpu_count() 
      logging.info('num_threads as multiprocessing.cpu_count', num_threads)
    except Exception:
      num_threads = 12
      logging.info('num_threads set by default', num_threads)
  #num_threads = 12

  if 'batch_size' in inspect.getargspec(decode_fn).args:
    decode_fn_ = decode_fn
    def decode_function(example):
      return decode_fn_(example, batch_size)
    decode_fn = decode_function

  if simple_parse and training:
    # for multiple gpu horovod run seem this much better, might due to repeat then shuffle better TODO 
    d = Dataset(files)
    if use_horovod and hvd_shard:
      d = shard(d)
    d = d.repeat(num_epochs).shuffle(batch_size * 1024).batch(batch_size).map(decode_fn, num_parallel_calls=num_threads).prefetch(9)
    return d.make_one_shot_iterator()
    
  if not min_after_dequeue: 
    min_after_dequeue = melt.tfrecords.read.MIN_AFTER_QUEUE

  if not num_epochs: 
    num_epochs = None

  if fix_random:
    if seed is None:
      seed = 1024
    shuffle_files = True  
    batch_join = False  #check can be True ?

    shuffle_batch = True
    #num_threads = 1

  if fix_sequence:
    no_random = True 
    allow_smaller_final_batch = True
    #num_threads = 1

  if no_random:
    shuffle_files = False
    batch_join = False
    shuffle_batch = False 

  drop_remainder = False if allow_smaller_final_batch else True
  #drop_remainder = True

  if not num_prefetch_batches:
    num_prefetch_batches = num_threads + 3
  
  if buffer_size is None:
    # ... Too small ? but 1024 will cause starup slow
    buffer_size = min_after_dequeue + num_prefetch_batches * batch_size
    #buffer_size = 1024 * batch_size
    
  with tf.name_scope(name):
    # https://github.com/tensorflow/tensorflow/issues/14857
    Dataset = Dataset or tf.data.TFRecordDataset
    if not shuffle_files or len(files) == 1:
      d = Dataset(files)
      if use_horovod and hvd_shard:
        d = shard(d)
    else:
      d = tf.data.Dataset.list_files(files)
      # here shard by files, not work good, especially for text line dataset with hrovod
      if use_horovod and shard_by_files:
        d = shard(d)
      if shuffle_files:
        d = d.shuffle(len(files), seed=seed)
        d = d.interleave(Dataset,
                        cycle_length=num_threads, block_length=1)
      if use_horovod and not shard_by_files:
        d = shard(d)

      #repeat_then_shuffle = True
      ## below on tf doc, but shard by files will cause problem, especially horovod mutlitple gpu, still not fully understand
      # Be sure to shard before you use any randomizing operator (such as shuffle).
      # Generally it is best if the shard operator is used early in the d pipeline. For example, when reading from a set of TFRecord files, shard before converting the d to input samples. This avoids reading every file on every worker. The following is an example of an efficient sharding strategy within a complete pipeline:
      # d = Dataset.list_files(pattern)
      # d = d.shard(num_workers, worker_index)
      # d = d.repeat(num_epochs)
      # d = d.shuffle(shuffle_buffer_size)
      # d = d.interleave(tf.data.TFRecordDataset,
      #                  cycle_length=num_readers, block_length=1)
      # d = d.map(parser_fn, num_parallel_calls=num_map_threads)

      # # # TODO still need shuffle here ?
      # # #https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
      # d = d.shuffle(num_files)\
      #                   .apply(tf.contrib.data.parallel_interleave(
      #                         Dataset, 
      #                         cycle_length=num_threads))

    if repeat and repeat_then_shuffle:
      d = d.repeat(num_epochs)

    # must batch then map if use pyfunc which you might use batch_parse, here batch_parse means batch parse otherwise slower but simple and powerfull...
    if not batch_parse:
      d = d.map(decode_fn, num_parallel_calls=num_threads)

    try:
      #shapes = d._output_shapes 
      shapes = tf.data.get_output_shapes(d)
    except Exception:
      shapes = None
    
    #logging.info('datast decode shapes', shapes)
    
    ## Has bug.. seems as least not work with bucket not sure without bucket ok or not
    if balance_pos_neg:
      # https://stackoverflow.com/questions/46938530/produce-balanced-mini-batch-with-d-api/49283371#49283371
      ds_pos = d.filter(pos_filter_fn).repeat()
      ds_neg = d.filter(neg_filter_fn)

      # def _concat(x, y):
      #   return tf.cond(tf.random_uniform(()) > 0.5, lambda: x, lambda: y)
      # d = tf.data.Dataset.zip((ds_pos, ds_neg))
      # d = d.map(_concat)

      d = tf.data.Dataset.zip((ds_pos, ds_neg))
      # Each input element will be converted into a two-element `Dataset` using
      # `Dataset.from_tensors()` and `Dataset.concatenate()`, then `Dataset.flat_map()`
      # will flatten the resulting `Dataset`s into a single `Dataset`.
      d = d.flat_map(
          lambda ex_pos, ex_neg: tf.data.Dataset.from_tensors(ex_pos).concatenate(
              tf.data.Dataset.from_tensors(ex_neg)))

    #https://github.com/tensorflow/tensorflow/issues/14451
    # count_fn for over sample
    if count_fn is not None:
      d = d.flat_map(
        lambda x, y : tf.data.Dataset.from_tensors((x, y)).repeat(tf.to_int64(count_fn(x, y))))

    # filter fn for under sample
    # if under_sample_filter_fn is not None:
    #   d = d.filter(under_sample_filter_fn)
      
    if filter_fn is not None:
      d = d.filter(filter_fn)
  
    if shuffle_batch:
      logging.info('shuffle with buffer_size', buffer_size, 'seed', seed)
      d = d.shuffle(buffer_size=buffer_size, seed=seed)

    # shuffle then repeat
    if repeat and not repeat_then_shuffle:
      d = d.repeat(num_epochs)

    # d = d.interleave(Dataset,
    #                       cycle_length=num_threads, block_length=16)


    # https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-d-map-d-prefetch-and-d-shuffle
    #d = d.prefetch(buffer_size)
    d = d.prefetch(num_prefetch_batches * batch_size)
    #d = d.prefetch(num_prefetch_batches)

    # #https://github.com/HKUST-KnowComp/R-Net/blob/master/util.py
    # #https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/data_reader.py
    if bucket_boundaries:
      # TODO remove support for length index, use use length key!
      assert length_key is not None or length_index is not None, 'forget to set length key  or length index ?'
      if not isinstance(bucket_boundaries, (list, tuple)):
        boundaries = [int(x) for x in bucket_boundaries.split(',') if x.strip()]
      else:
        boundaries = bucket_boundaries
      logging.info('bucket_boundaries', boundaries)
      with tf.name_scope("bucket_by_seq_length"):
        def example_to_bucket_id(*args, **kw):
          """Return int64 id of the length bucket for this example."""
          #assert length_index is not None
          if length_key is None:
            try:
              x = list(args[0])[length_index]
            except Exception:
              x = args[length_index]
          else:
            try:
              x = args[0][length_key]
            except Exception:
              x = args[length_key]      
          
          seq_length = tf.reduce_sum(tf.cast(tf.cast(x, tf.bool), tf.int32))
          
          buckets_min = [np.iinfo(np.int32).min] + boundaries
          buckets_max = boundaries + [np.iinfo(np.int32).max]
          conditions_c = tf.logical_and(
              tf.less_equal(buckets_min, seq_length),
              tf.less(seq_length, buckets_max))
          bucket_id = tf.reduce_min(tf.where(conditions_c))
          return bucket_id

        if not bucket_batch_sizes:
          def batching_fn(bucket_id, grouped_d):
              return grouped_d.padded_batch(batch_size, padded_shapes=(shapes))

          ## TODO larger window better hsku squad doing this like below, shuffle can be better ?
          ## NOTICE!! shuffle may be slow start fill queue can remove not hurt performance ?
          d = d.apply(tf.contrib.data.group_by_window(
            example_to_bucket_id, batching_fn, window_size=5 * batch_size)).shuffle((len(boundaries) + 1) * 25)

          ## tenor2tensor doing this, no shuffle ? also it seems like window_func for different bounds
          ## with different batch_size ?
          # d = d.apply(
          #   tf.contrib.data.group_by_window(example_to_bucket_id, batching_fn, batch_size)).shuffle((len(boundaries) + 1) * 25)
        else:
          # TEST OK 
          # test ok ie buckets[400] batch_sizes[64, 32]
          if not isinstance(bucket_batch_sizes, (list, tuple)):
            bucket_batch_sizes = [int(x) for x in bucket_batch_sizes.split(',') if x.strip()]

          logging.info('bucket_batche_sizes', bucket_batch_sizes)
          assert len(boundaries) + 1 == len(bucket_batch_sizes)

          def window_size_fn(bucket_id):
            # window size = batch size
            batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)
            window_size = batch_sizes[bucket_id]
            # * 5 will make reading slower
            window_size *= 5
            return window_size

          def batching_fn(bucket_id, grouped_d):
            batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)
            batch_size = batch_sizes[bucket_id]
            #return padded_batch(grouped_d, batch_size, padded_shapes=None)
            return grouped_d.padded_batch(batch_size, padded_shapes=(shapes))

          # shuffle will make start slower might fill
          d = d.apply(tf.contrib.data.group_by_window(
            example_to_bucket_id, batching_fn, None, window_size_fn)).shuffle((len(boundaries) + 1) * 25)      
    else:
      # no bucket
      if dynamic_pad and not batch_parse:
        d = d.padded_batch(batch_size, padded_shapes=(shapes), drop_remainder=drop_remainder)
      else:
        d = d.batch(batch_size, drop_remainder=drop_remainder)
        if batch_parse:
          d = d.map(decode_fn, num_parallel_calls=num_threads)

    # if not allow_smaller_final_batch:
    #   # https://github.com/tensorflow/tensorflow/issues/13745 d.apply(tf.contrib.data.batch_and_drop_remainder(10)).
    #   d = d.filter(lambda x, *args, **kw: tf.equal(tf.shape(x)[0], batch_size))

  # TODO save iterator ?
  ## Create saveable object from iterator.
  #saveable = tf.contrib.data.make_saveable_from_iterator(iterator)

  # Save the iterator state by adding it to the saveable objects collection.
  #tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
    #try:
    if tf.executing_eagerly():
      # TODO store iterator for eager
      return d
    else:
      if repeat and not initializable:
        iterator = d.make_one_shot_iterator() 
        saveable = tf.contrib.data.make_saveable_from_iterator(iterator)
        tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
        if return_iterator:
          return iterator
        ops = iterator.get_next()
        return ops
      else:
        iterator = d.make_initializable_iterator()
        saveable = tf.contrib.data.make_saveable_from_iterator(iterator)
        tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
        return iterator
    # except Exception:
    #   if repeat and not initializable:
    #     iterator = d.make_one_shot_iterator()
    #     saveable = tf.contrib.data.make_saveable_from_iterator(iterator)
    #     tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
    #     if return_iterator:
    #       return iterator
    #     ops = iterator.get_next()
    #     return ops
    #   else:
    #     # if not repeat then need to init iterator each epoch
    #     iterator = d.make_initializable_iterator()
    #     saveable = tf.contrib.data.make_saveable_from_iterator(iterator)
    #     tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
    #     return iterator         
