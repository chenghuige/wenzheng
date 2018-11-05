#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   input_flags.py
#        \author   chenghuige  
#          \date   2016-12-25 00:17:18.268341
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
#import gflags as flags
FLAGS = flags.FLAGS
  
#-------- train
flags.DEFINE_integer('min_records', 0, '')
flags.DEFINE_integer('num_records', 0, 'if not 0, will check equal')

#--------- read data
flags.DEFINE_integer('fixed_eval_batch_size', 30, """must >= num_fixed_evaluate_examples
                                                     if == real dataset len then fix sequence show
                                                     if not == can be show different fixed each time
                                                     usefull if you want only show see 2 and 
                                                     be different each time
                                                     if you want see 2 by 2 seq
                                                     then num_fixed_evaluate_example = 2
                                                          fixed_eval_batch_size = 2
                                                  """)

flags.DEFINE_integer('num_fixed_evaluate_examples', 30, '')
flags.DEFINE_integer('num_evaluate_examples', 1, '')

#flags.DEFINE_integer('num_threads', 12, """threads for reading input tfrecords,
#                                           setting to 1 may be faster but less randomness
#                                        """)

flags.DEFINE_boolean('shuffle_files', True, '')
flags.DEFINE_boolean('batch_join', True, '')
flags.DEFINE_boolean('shuffle_batch', True, '')

flags.DEFINE_boolean('shuffle_then_decode', True, 
                     """ actually this is decided by is_sequence_example.. 
                     if is_sequence_example then False, if just example not sequence then True since is sparse
                     TODO remove this
                     """)
flags.DEFINE_boolean('is_sequence_example', False, '')

flags.DEFINE_boolean('dynamic_batch_length', True, 
                     """very important False means all batch same size! 
                        otherwise use dynamic batch size
                        Now only not sequence_example data will support dyanmic_batch_length=False
                        Also for cnn you might need to set to False to make all equal length batch used
                        """)

flags.DEFINE_boolean('use_weights', False, '''from tfrecord per example word, usaually tf*idf weight, 
                                              same word has difference score in different example/instances''')
flags.DEFINE_string('weights', None, '')
flags.DEFINE_boolean('use_idf_weights', False, 'idf only weight vocab based fixed for each vocab word')

flags.DEFINE_boolean('use_inst_weights', False, 'use per instantce weights')
  
flags.DEFINE_boolean('feed_dict', False, 'depreciated, too complex, just prepare your data at first for simple')


#--for scene
flags.DEFINE_string('scene_train_input', None, '')
flags.DEFINE_string('scene_valid_input', None, '')
flags.DEFINE_boolean('use_scene_embedding', True, '')

#----------eval
flags.DEFINE_boolean('legacy_rnn_decoder', False, '')
flags.DEFINE_boolean('experiment_rnn_decoder', False, '')

#----------strategy 

flags.DEFINE_string('seg_method', 'basic', '')
#flags.DEFINE_boolean('feed_single', False, '')

flags.DEFINE_boolean('gen_predict', True, '')


flags.DEFINE_string('decode_name', 'text', '')
flags.DEFINE_string('decode_str_name', 'text_str', '')

flags.DEFINE_boolean('reinforcement_learning', False, '')
flags.DEFINE_float('reinforcement_ratio', 1., '')

#--------for image caption  TODO move to image_caption/input_flags.py ?
#--if use image dir already info in image_features
flags.DEFINE_string('image_dir', None, 'input images dir')

flags.DEFINE_boolean('pre_calc_image_feature', False, 'will set to true if not has image model auto in train.py')
flags.DEFINE_boolean('has_image_model', False, '')

flags.DEFINE_string('image_checkpoint_file', '/home/gezi/data/image_model_check_point/inception_resnet_v2_2016_08_30.ckpt', '''None means image model from scratch''')
flags.DEFINE_boolean('finetune_image_model', True, '''by default will be finetune otherwise
                                                   why not pre calc image feature much faster
                                                   but we also support''')
flags.DEFINE_float('image_lr_ratio', 0.1, '')

flags.DEFINE_string('finetune_end_point', None, 'if not None, only finetune from some ende point layers before will freeze')
                                                   
flags.DEFINE_boolean('distort_image', True, 'training option')
flags.DEFINE_boolean('random_crop_image', True, 'training option')

flags.DEFINE_string('image_model_name', None, 'InceptionResnetV2 might be by default but can get from checkpoint direclty')
flags.DEFINE_string('image_endpoint_feature_name', None, 'mostly None for showandtell, not None for show attend and tell features if not in endpoint dict')
flags.DEFINE_integer('image_attention_size', None, 'InceptionResnetV2 will be 64')
flags.DEFINE_integer('image_feature_len', None, '')
flags.DEFINE_integer('image_feature_len_decode', None, 'if image_feature len means final output image_feature_len then use image_feature_len_input as input image feature len')
flags.DEFINE_integer('image_width', None, 'default width of inception 299, resnet 224 but google pretrain resnet v2 models also 299')
flags.DEFINE_integer('image_height', None, 'default height of inception 299, resnet 224 but google pretrain resnet v2 models also 299')
flags.DEFINE_boolean('image_features2feature', False, '''for show and tell if input is pre calc atteniton features input
                                                                    here set True will process attention features 
                                                                    to generate and use final feature jus similar like input without 
                                                                    attention vectors''')

# for image classification
flags.DEFINE_integer('num_image_classes', None, '')
flags.DEFINE_integer('num_pretrain_image_classes', None, 'HACK for using pretrain image models where class num is not 1001 as imagenet 1k label models')
flags.DEFINE_integer('image_top_k', 3, '')

flags.DEFINE_string('scene_model', None, 'if not None will use scene_model otherwise it is scene_model path')
flags.DEFINE_string('scene_cats', '/home/gezi/mine/hasky/deepiu/scene/place365/cat_names_cn.txt', '')
flags.DEFINE_integer('scene_feature_len', 15, '')

                                                  
#---in melt.apps.image_processing.py
#flags.DEFINE_string('image_model_name', 'InceptionV3', '')
flags.DEFINE_string('one_image', '/home/gezi/data/flickr/flickr30k-images/1000092795.jpg', '')

flags.DEFINE_string('image_feature_name', 'image_feature', 'for decoding tfrecord')


#---------negative smapling
flags.DEFINE_integer('num_negs', 1, '0 means no neg')
flags.DEFINE_integer('num_eval_negs', 1, '0 means no neg')
flags.DEFINE_boolean('neg_left', False, 'ltext or image')
flags.DEFINE_boolean('neg_right', True, 'rtext or text')
flags.DEFINE_boolean('neg_correct_ratio', True, '')


#---------discriminant trainer

flags.DEFINE_string('activation', 'relu', 
                    """relu/tanh/sigmoid  seems sigmoid will not work here not convergent
                    and relu slightly better than tanh and convrgence speed faster""")
flags.DEFINE_boolean('bias', False, 'wether to use bias. Not using bias can speedup a bit')

flags.DEFINE_boolean('elementwise_predict', False, '')


flags.DEFINE_float('keep_prob', 1., 'or 0.9 0.8 0.5')
flags.DEFINE_float('dropout', 0., 'or 0.9 0.8 0.5')

flags.DEFINE_string('trainer_scope', None, '')


#----- encoder 
flags.DEFINE_string('encoder_type', None, '')
flags.DEFINE_string('image_encoder', 'ShowAndTell', '')
flags.DEFINE_string('text_encoder', None, '')
flags.DEFINE_string('mlp_dims', None, 'like 512 128,32')
flags.DEFINE_float('mlp_keep_prob', 1., '')
flags.DEFINE_string('encoder_output_method', 'max', '')

flags.DEFINE_integer('top_k', 3, '')

# text
flags.DEFINE_bool('add_start_end', False, '')

#----- cnn  TODO
flags.DEFINE_integer('num_filters', 128, '')

#----- other
flags.DEFINE_float('label_smoothing', 0, '')

flags.DEFINE_integer('finetune_emb_step', None, 'might be 45000 toxic 20 epoch')


flags.DEFINE_bool('mask_pooling', True, '')
flags.DEFINE_integer('hop', 2, '')
flags.DEFINE_integer('label_hop', 1, '')

#--------------------
flags.DEFINE_integer('num_finetune_words', None, '')
flags.DEFINE_integer('num_finetune_chars', None, '')

flags.DEFINE_string('lm_path', None, '')
flags.DEFINE_float('lm_lr_factor', 1., '')
flags.DEFINE_bool('lm_model', False, '')
flags.DEFINE_bool('dynamic_finetune', False, '')
flags.DEFINE_string('char_encoder', 'rnn', '')

flags.DEFINE_bool('use_char_emb', True, 'if use char and not use char emb then use another char emb different from emb other wise share to use one emb')
flags.DEFINE_bool('use_simple_ngrams', False, '')
flags.DEFINE_string('char_combiner', 'concat', '')
flags.DEFINE_bool('char_padding', False, '')

flags.DEFINE_bool('use_ngrams', False, '')
flags.DEFINE_bool('use_fngrams', False, '')
flags.DEFINE_integer('ngram_emb_dim', 300, '')
flags.DEFINE_string('ngram_combiner', 'sum', 'sum or concat or dsfu')
flags.DEFINE_string('ngram_self_combiner', 'sum', 'sum or concat')
flags.DEFINE_bool('ngram_only', False, '')