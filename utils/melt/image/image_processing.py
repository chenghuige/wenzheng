# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Copy from google im2txt
Helper functions for image preprocessing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
#from tensorflow.python.ops import variable

import tensorflow as tf

import tensorflow.contrib.slim as slim
#TODO must set PYTHON_PATH for models/slim
try:
  from preprocessing import preprocessing_factory
  from nets import nets_factory
except Exception:
  pass
#from melt.slim import base_nets_factory 
import gezi
import melt
from melt import logging

import sys
import os
  
def read_image(image_path):
  #with tf.device('/cpu:0'):
  # https://blog.csdn.net/u013555719/article/details/77991010 change to use rb for python3
  with tf.gfile.FastGFile(image_path, "rb") as f:
    encoded_image = f.read()
  return encoded_image

def get_imagenet_from_checkpoint(checkpoint_path):
  """
  net = get_net_from_checkpoint(checkpoint)
  net.func_name  # like inception_v4
  net.default_image_size # like 299
  """
  checkpoint = melt.get_model_path(checkpoint_path)
  if not checkpoint or \
   (not os.path.exists(checkpoint) \
      and not os.path.exists(checkpoint + '.index')):
    return None 
    
  from tensorflow.python import pywrap_tensorflow 
  reader = pywrap_tensorflow.NewCheckpointReader(checkpoint)
  var_to_shape_map = reader.get_variable_to_shape_map()
  name = None
  for key in var_to_shape_map.keys():
    name = key.split('/')[0]
    gnu_name = gezi.to_gnu_name(name)
    if gnu_name in nets_factory.networks_map:
      break
  if name is None:
    return None 
  else:
    nets_factory.networks_map[gnu_name].name = name
    return nets_factory.networks_map[gnu_name]

def get_net_from_checkpoint(checkpoint):
  """
  net = get_net_from_checkpoint(checkpoint)
  net.func_name  # like inception_v4
  net.default_image_size # like 299
  """
  from tensorflow.python import pywrap_tensorflow 
  reader = pywrap_tensorflow.NewCheckpointReader(checkpoint)
  var_to_shape_map = reader.get_variable_to_shape_map()
  name = None
  for key in var_to_shape_map.keys():
    name = key.split('/')[0]
    gnu_name = gezi.to_gnu_name(name)
    if gnu_name in nets_factory.networks_map:
      break
  if name is None:
    return None 
  else:
    return nets_factory.networks_map[name], name

def create_image_model_init_fn(image_model_name, image_checkpoint_file, 
                               moving_average_decay=None):
  """
  NOTICE whey moving_average_decay is not None will add variables
  so for finetune image model you may need to escape scope! 
  see train.py of image_caption
  """
  # print(tf.all_variables())
  # print(tf.get_default_graph().get_all_collection_keys())
  # refer to slim eval_image_classifer.py
  
  # HACK for nasnet, also for nasnet you need to use 
  # rename-varaibles.py --checkpoint_dir=... --add_prefix=nasnet_large to add global scope 
  # and use the converted model 
  variables_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=image_model_name)

  assert variables_to_restore, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

  if image_model_name.startswith('nasnet') and not moving_average_decay:
      moving_average_decay = 0.9999

  print('in create_image_model_init_fn image_model_name', image_model_name, 'moving_average_decay', moving_average_decay, file=sys.stderr)

  # TODO understand moving average decay and it's effect on graph...
  if moving_average_decay:   
    with tf.variable_scope(image_model_name):
      tf_global_step = tf.train.get_or_create_global_step()
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, tf_global_step)
    #variables_to_restore = variable_averages.variables_to_restore(slim.get_model_variables())
    variables_to_restore = variable_averages.variables_to_restore(variables_to_restore)
    # well still has nont image_model_name start vars ... TODO
    # TODO for finetune final pre_logits feature or classification need to exclude global_step other wise not found global step realated moving.. vars ..
    # TODO need to check ignore global step will not affect caption model using nasnet
    variables_to_restore = dict([(key, val) for key, val in variables_to_restore.items() if key.startswith(image_model_name) and not 'global_step' in key])
    #variables_to_restore = dict([(key, val) for key, val in variables_to_restore.items() if key.startswith(image_model_name)])

    # # #TODO just hack right now, we must use another like InceptionV3 to construct InceptionV3/InceptionV3
    # #not for escaple scope, can just with scope('') but for reuse.. when predict TODO..
    # def name_in_checkpoint(var):
    #   return var.op.name.replace('/' + image_model_name, '')
    # variables_to_restore = {name_in_checkpoint(var):var for var in variables_to_restore}

  saver = tf.train.Saver(variables_to_restore)

  def restore_fn(sess):
    timer = gezi.Timer('restore image var from %s %s' % (image_model_name, image_checkpoint_file))
    logging.info("Restoring image variables from checkpoint file %s" % image_checkpoint_file)
    saver.restore(sess, image_checkpoint_file)
    timer.print()

  return restore_fn

def distort_image(image, distort_color=True):
  """Perform random distortions on an image.

  Args:
    image: A float32 Tensor of shape [height, width, 3] with values in [0, 1).

  Returns:
    distorted_image: A float32 Tensor of shape [height, width, 3] with values in
      [0, 1].
  """
  # Randomly flip horizontally.
  with tf.name_scope("flip_horizontal", values=[image]):
    image = tf.image.random_flip_left_right(image)

  if distort_color:
    # Randomly distort the colors based on thread id.
    with tf.name_scope("distort_color", values=[image]):
      def _distort_image_fn1(image):
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.032)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        return image 

      def _distort_image_fn2(image):
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.032)
        return image 

      p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
      pred = tf.less(p_order, 0.5)
      image = tf.cond(pred, lambda: _distort_image_fn1(image), lambda: _distort_image_fn2(image))

      # The random_* ops do not necessarily clamp.
      image = tf.clip_by_value(image, 0.0, 1.0)

  return image

# #--depreciated just use tf ones
# def decode_image(contents, channels=None, name=None):
#   """Convenience function for `decode_gif`, `decode_jpeg`, and `decode_png`.
#   Detects whether an image is a GIF, JPEG, or PNG, and performs the appropriate 
#   operation to convert the input bytes `string` into a `Tensor` of type `uint8`.

#   Note: `decode_gif` returns a 4-D array `[num_frames, height, width, 3]`, as 
#   opposed to `decode_jpeg` and `decode_png`, which return 3-D arrays 
#   `[height, width, num_channels]`. Make sure to take this into account when 
#   constructing your graph if you are intermixing GIF files with JPEG and/or PNG 
#   files.

#   Args:
#     contents: 0-D `string`. The encoded image bytes.
#     channels: An optional `int`. Defaults to `0`. Number of color channels for 
#       the decoded image.
#     name: A name for the operation (optional)
    
#   Returns:
#     `Tensor` with type `uint8` with shape `[height, width, num_channels]` for 
#       JPEG and PNG images and shape `[num_frames, height, width, 3]` for GIF 
#       images.
#   """
#   with ops.name_scope(name, 'decode_image') as scope: 
#     #return gen_image_ops.decode_jpeg(contents, channels)
#     if channels not in (None, 0, 1, 3):
#       raise ValueError('channels must be in (None, 0, 1, 3)')
#     #substr = string_ops.substr(contents, 0, 4)
#     substr = string_ops.substr(contents, 0, 1)

#     def _gif():
#       raise ValueError('not jpeg or png')
#       # Create assert op to check that bytes are GIF decodable
#       #is_gif = math_ops.equal(substr, b'\x47\x49\x46\x38', name='is_gif')
#       is_gif = math_ops.equal(substr, b'\x47', name='is_gif')

#       decode_msg = 'Unable to decode bytes as JPEG, PNG, or GIF'
#       assert_decode = control_flow_ops.Assert(is_gif, [decode_msg])
#       # Create assert to make sure that channels is not set to 1
#       # Already checked above that channels is in (None, 0, 1, 3)
#       gif_channels = 0 if channels is None else channels
#       good_channels = math_ops.not_equal(gif_channels, 1, name='check_channels')
#       channels_msg = 'Channels must be in (None, 0, 3) when decoding GIF images'
#       assert_channels = control_flow_ops.Assert(good_channels, [channels_msg])
#       with ops.control_dependencies([assert_decode, assert_channels]):
#         return gen_image_ops.decode_gif(contents)
    
#     def _png():
#       return gen_image_ops.decode_png(contents, channels)
    
#     def check_png():
#       #is_png = math_ops.equal(substr, b'\211PNG', name='is_png')
#       is_png = math_ops.equal(substr, b'\x89', name='is_png')
#       return control_flow_ops.cond(is_png, _png, _gif, name='cond_png')
    
#     def _jpeg():
#       return gen_image_ops.decode_jpeg(contents, channels)

#     #is_jpeg = math_ops.equal(substr, b'\xff\xd8\xff\xe0', name='is_jpeg')
#     is_jpeg = math_ops.equal(substr, b'\xff', name='is_jpeg')
#     try:
#       return control_flow_ops.cond(is_jpeg, _jpeg, check_png, name='cond_jpeg')
#     except Exception:
#       #this is unsafe, since might not be jpeg format will raise error in c++ code, unable to catch
#       return gen_image_ops.decode_jpeg(contents, channels, try_recover_truncated=True, acceptable_fraction=10)

def decode_image(contents, channels=3, image_format='jpeg', dtype=None):
  with tf.name_scope("decode", values=[contents]):
    if image_format == "jpeg":
      #---TODO this will cause hang.... try_recover_truncated=True, acceptable_fraction=10 will hang...
      #image = tf.image.decode_jpeg(contents, channels=3, try_recover_truncated=True, acceptable_fraction=10)
      image = tf.image.decode_jpeg(contents, channels=channels)
    elif image_format == "png":
      image = tf.image.decode_png(contents, channels=channels)
    else:
      #--why here will casue no size.. might for gif..., for safe just use image_format jpeg? TODO FIXME
      image = tf.image.decode_image(contents, channels=3)
    if dtype:
      image = tf.image.convert_image_dtype(image, dtype=dtype)
    return image

def process_image(encoded_image,
                  is_training,
                  height,
                  width,
                  resize_height=346,
                  resize_width=346,
                  random_crop=True,
                  distort=True,
                  distort_color=True,
                  image_format=None):
  """Decode an image, resize and apply random distortions.

  In training, images are distorted slightly differently depending on thread_id.

  Args:
    encoded_image: String Tensor containing the image.
    is_training: Boolean; whether preprocessing for training or eval.
    height: Height of the output image.
    width: Width of the output image.
    resize_height: If > 0, resize height before crop to final dimensions.
    resize_width: If > 0, resize width before crop to final dimensions.
    thread_id: Preprocessing thread id used to select the ordering of color
      distortions. There should be a multiple of 2 preprocessing threads.
    image_format: "jpeg" or "png".

  Returns:
    A float32 Tensor of shape [height, width, 3] with values in [-1, 1].

  Raises:
    ValueError: If image_format is invalid.
  """
  # Helper function to log an image summary to the visualizer. Summaries are
  # only logged in thread 0.  TODO summary
  def image_summary(name, image):
    tf.summary.image(name, tf.expand_dims(image, 0))

  # Decode image into a float32 Tensor of shape [?, ?, 3] with values in [0, 1).
  image = decode_image(encoded_image, channels=3, image_format=image_format, dtype=tf.float32)

  #TODO summary
  #image_summary("original_image", image)

  # Resize image.
  assert (resize_height > 0) == (resize_width > 0)
  if resize_height:
    image = tf.image.resize_images(image,
                                   size=[resize_height, resize_width],
                                   method=tf.image.ResizeMethod.BILINEAR)

  # Crop to final dimensions.
  if is_training and random_crop:
    image = tf.random_crop(image, [height, width, 3])
  else:
    # Central crop, assuming resize_height > height, resize_width > width.
    image = tf.image.resize_image_with_crop_or_pad(image, height, width)
  
  #image_summary("resized_image", image)

  # Randomly distort the image.
  if is_training and distort:
    image = distort_image(image, distort_color)
  
  #image_summary("final_image", image)

  # Rescale to [-1,1] instead of [0, 1]
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image


import melt


# Depreciated! 
def create_image2feature_fn(name='InceptionV3'):
  """
  will be depreciated just use create_image2feature_slim_fn 
  """
  #NOTICE how this method solve run/ scope problem, scope must before def, learn this from melt.seq2seq.attention_decoder_fn.py
  raise TypeError('please use create_image2feature_slim_fn')
  with tf.variable_scope(name) as scope:
    def construct_fn(encoded_image, 
                     height, 
                     width, 
                     trainable=False,
                     is_training=False,
                     resize_height=346,
                     resize_width=346,
                     distort=True,
                     image_format="jpeg",  #for safe just use decode_jpeg
                     reuse=None):
      #decode image .. only accept one image so use map_fn, but seems map_fn is slow, so use as less as possible
      #here decode image + preprocess into one
      image = tf.map_fn(lambda img: process_image(img,
                                                  is_training=is_training,
                                                  height=height, 
                                                  width=width,
                                                  resize_height=resize_height,
                                                  resize_width=resize_width,
                                                  distort=distort,
                                                  image_format=image_format), 
                                encoded_image,
                                dtype=tf.float32)


      #this is the same as slim for inception v3, and TODO can be modified to accept other models
      image_feature = melt.image.image_embedding.inception_v3(
        image,
        trainable=trainable,
        is_training=is_training,
        reuse=reuse,
        scope=scope)

      #if not set this eval_loss = trainer.build_train_graph(eval_image_feature, eval_text, eval_neg_text) will fail
      #but still need to set reuse for melt.image.image_embedding.inception_v3... confused.., anyway now works..
      #with out reuse=True score = predictor.init_predict() will fail, resue_variables not work for it..
      #trainer create function once use it second time(same function) work here(with scope.reuse_variables)
      #predictor create another function, though seem same name same scope, but you need to set reuse=True again!
      #even if use tf.make_template still need this..
      scope.reuse_variables()
      return image_feature

    return construct_fn


info = {
  'vgg_19': { 
              'feature_dim': 512,
              'num_features': 196 
            },

  'InceptionV3': { 
                  'feature_dim': 2048 ,
                  'width': 299,
                  'height': 299
                 },

  'InceptionResnetV2': { 
                            'feature_dim': 1536,
                            'feature_end_point': 'PreLogitsFlatten',
                            'num_features': 64,  
                            'features_end_point': 'Conv2d_7b_1x1',
                            'width': 299,
                            'height': 299
                        },

  'InceptionV4': {
                    'feature_dim': 1536,
                    'feature_end_point': 'PreLogitsFlatten',
                    'num_features': 64, 
                    'features_end_point': 'Mixed_7d',
                    'width': 299,
                    'height': 299
                 },
  # TODO may be group as resnet_v2
  'resnet_v2_152': {
    'feature_dim': 2048,
    'feature_end_point': 'PreLogits',
    'num_features': 100,
    'features_end_point': 'attention',
    'width': 299,
    'height': 299
  },
  'resnet_v1_101': {
    'feature_dim': 2048,
    'feature_end_point': 'PreLogits',
    'num_features': 100,
    'features_end_point': 'attention',
    'width': 299,
    'height': 299    
  },
  'nasnet_large': {
    'feature_dim': 4032,
    'feature_end_point': 'PreLogits',
    'num_features': 121,
    'features_end_point': 'attention',
    'width': 331,
    'height': 331
  },
  'nasnet_v1_large': {
    'feature_dim': 4032,
    'feature_end_point': 'PreLogits',
    'num_features': 121,
    'features_end_point': 'attention',
    'width': 331,
    'height': 331
  },
  'nasnet_v2_large': {
    'feature_dim': 4032,
    'feature_end_point': 'PreLogits',
    'num_features': 121,
    'features_end_point': 'attention',
    'width': 331,
    'height': 331
  },
  'nasnet_mobile': {
    'feature_dim': 1056,
    'feature_end_point': 'PreLogits',
    'num_features': 49,
    'features_end_point': 'attention',
    'width': 224,
    'height': 224
  }
}

def get_features_name(image_model_name):
  return info[image_model_name]['features_end_point']

def get_num_features(image_model_name):
  return info[image_model_name]['num_features']

def get_feature_dim(image_model_name):
  return info[image_model_name]['feature_dim']

def OpenimageV2PreprocessImage(image, network='resnet_v1_101', image_size=299, is_training=False):
  # If resolution is larger than 224 we need to adjust some internal resizing
  # parameters for vgg preprocessing.
  if any(network.startswith(x) for x in ['resnet', 'vgg']):
    preprocessing_kwargs = {
        'resize_side_min': int(256 * image_size / 224),
        'resize_side_max': int(512 * image_size / 224)
    }
  else:
    preprocessing_kwargs = {}
  preprocessing_fn = preprocessing_factory.get_preprocessing(
      name=network, is_training=is_training)

  height = image_size
  width = image_size
  image = preprocessing_fn(image, height, width, **preprocessing_kwargs)
  image.set_shape([height, width, 3])
  return image

def create_image2feature_slim_fn(name=None, 
                                 feature_name=None, 
                                 preprocess_image=True, 
                                 num_classes=None,
                                 scope=''):
  """
    #NOTICE how this method solve run/ scope problem, scope must before def
    using slim util to create image feature
    You need to set PythonPath to include models/slim or install it if using on hdfs run TODO check
  """
  #TODO '' can not work to be used as to escope scope ..., so if want to escape and call this function 
  #anywhere need to modify slim code to let slim code with scope None '' instead of name here like InceptionV3 
  #HACK add another socpe.. to escape.. see hasky/jupter/scope.ipynb

  ##notice for vgg.. scope is also in gnu format unlike IncepitonV3 -> inception_v3 or like resnet_v2_152 scope and model name all in gnu format
  ##ValueError: Can not squeeze dim[1], expected a dimension of 1, got 3 for 'bow/main/image_text_sim/vgg_19/fc8/squeezed' (op: 'Squeeze') with input shapes: [32,3,3,1001].
  #name = 'vgg_19' 
  
  # HACK for nasnet which not has a nasnet_large/ top scope, I have converted the model file to add this scope 
  # so here have also add this top scope  TODO whey still add.. show and tell top scope not escape ? Now workaround is change slim code, add top scope there
  #if name.startswith('nasnet') and not scope:
  #  scope = name 

  with tf.variable_scope(scope) as scope:
  # with tf.variable_scope(name) as scope: #name will be set in slim nets
    def construct_fn(encoded_image, 
                     height=None, 
                     width=None, 
                     trainable=False, 
                     is_training=False,
                     resize_height=346,
                     resize_width=346,
                     random_crop=True,
                     distort=True,
                     distort_color=True,
                     slim_preprocessing=True,  # if image_model_name.startswith('nasnet') and not moving_average_decay:
                     weight_decay=0.00004,
                     finetune_end_point=None,
                     feature_name=feature_name,
                     num_classes=num_classes,      
                     image_format="jpeg",  #for safe just use decode_jpeg
                     reuse=None): 
      logging.info('image model trainable:{}, is_training:{}'.format(trainable, is_training))

      #allow [batch_size, 1] as input
      #print(encoded_image.shape) #should be (?,)
      #encoded_image = tf.squeeze(encoded_image) #this will casue problem if input batch size is 1, so squeeze seems danerous TODO check
      #if use tf.squeeze(encoded_image, 1) also not work, out of index TODO can if len(shape) > 1 squeeze ?

      #below is alos ok? TODO CHECK
      # shape_list = encoded_image.get_shape().as_list()
      # if len(shape_list) > 1:
      #   encoded_image = tf.squeeze(encoded_image, -1)
      
      #preprocess_image
      assert name is not None
      net_name = gezi.to_gnu_name(name)

      height = height or info[net_name]['height']
      width = width or info[net_name]['width']

      #well this is slightly slow and the result is differnt from im2txt inceptionV3 usage result, 
      #use im2txt code seems ok, not sure if slim preprocess will be better! TODO
      #for inception related model I think im2txt process will be fine, for other models not sure TODO
      #using slim preprocessing real 2m45.737s user  3m12.896s sys 0m10.265s 
      #using im2txt processing real 2m46.709s user  3m8.067s sys 0m8.297s  
      #and the final feature will be slightly differnt 
      #one thing intersting is use 2 tf.map_fn(1 decode image, 1 preprocess) will be much slower then use 1 tf.map_fn (decode image+ preprocess)
      if preprocess_image:
        batch_size = encoded_image.get_shape()[0].value or tf.shape(encoded_image)[0]
        encoded_image = tf.reshape(encoded_image, [batch_size,])
        if slim_preprocessing:
          # HACK HERE
          # https://github.com/tensorflow/models/tree/master/research/slim
          # ^ ResNet V2 models use Inception pre-processing and input image size of 299 (use --preprocessing_name inception 
          # --eval_image_size 299 when using eval_image_classifier.py).
          # TODO FIXME alos seems vgg prprocessing will got similar feature for resnet_v2_152.. why?
          # TODO also for resnet_v1_101 using openimage pretrain model, must also use 299 if using 224 then generated feature is all same for images
          net_name_ = net_name 

          # HACK assume resnet101 model right now is only OpenimageV2 one  TODO
          if net_name == 'resnet_v1_101':
            print('HACK for resnet_v1_101 openimage checkpoint preprocess image', file=sys.stderr)
            # TODO FIXME !! Notice set dtype=tf.float32 will casue much diff in result for resnet 101 openimage
            # image = tf.map_fn(lambda img: OpenimageV2PreprocessImage(decode_image(img, image_format=image_format, dtype=tf.float32), is_training=(is_training and distort)),
            #         encoded_image, dtype=tf.float32)
            image = tf.map_fn(lambda img: OpenimageV2PreprocessImage(decode_image(img, image_format=image_format), is_training=(is_training and distort)),
                    encoded_image, dtype=tf.float32)
          else:
            if net_name.startswith('resnet_v2'):
            #if net_name.startswith('resnet'):
              net_name_ = 'inception'
              height = 299
              width = 299
              print('HACK here adjust to use inception preprocessing and inception default height and width', file=sys.stderr)

            preprocessing_fn = preprocessing_factory.get_preprocessing(net_name_, is_training=(is_training and distort))        

            print('preprocessing_fn net_name', net_name_, 'height', height, 'width', width, file=sys.stderr)
            assert height is not None
            # Iteresting for other models other then resnet101 openimage model, with or without dtype=tf.float32 is all ok
            #with tf.device('/cpu:0'):
            # image = tf.map_fn(lambda img: preprocessing_fn(decode_image(img, image_format=image_format, dtype=tf.float32), height, width),
            #                   encoded_image, dtype=tf.float32)
            # set add_image_summaries to False for tf1.5
            # otherwise alueError: Cannot use 'show_and_tell/main/encode/map/while/distort_image/image_with_bounding_boxes' as input to 'show_and_tell_1/Merge/MergeSummary' 
            # because 'show_and_tell/main/encode/map/while/distort_image/image_with_bounding_boxes' is in a while loop. See info log for more details.
            image = tf.map_fn(lambda img: preprocessing_fn(decode_image(img, image_format=image_format), height, width, add_image_summaries=False),
                              encoded_image, dtype=tf.float32)
        else:
          raise ValueError('not use im2txt stype preprocessing any more just use slim preprocessing')
          # im2txt style preprocessing
          #with tf.device('/cpu:0'):
          image = tf.map_fn(lambda img: process_image(img,
                                                    is_training=is_training,
                                                    height=height, 
                                                    width=width,
                                                    resize_height=resize_height,
                                                    resize_width=resize_width,
                                                    random_crop=random_crop,
                                                    distort=distort,
                                                    distort_color=distort_color,
                                                    image_format=image_format), 
                                                    encoded_image,
                                                    dtype=tf.float32)
      else:
        image = encoded_image

      # TODO like image_embedding.py add batch_norm ? fully understand!
      is_image_model_training = trainable and is_training
      if trainable and weight_decay:
        weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
      else:
        weights_regularizer = None

      with tf.variable_scope(scope, reuse=reuse):
        # TODO Remove this arg scope ? since net_factory get function with proper arg scope!
        with slim.arg_scope(
          [slim.conv2d, slim.fully_connected],
          weights_regularizer=weights_regularizer,
          trainable=trainable): #should this be faster then stop_gradient? exp show this slim.arg_scope with trainable=False work
     
          # actually final num class layer not used for image feature purpose, but since in check point train using 1001, for simplicity here set 1001
          # TODO might set try to use num_classes=None or 0, which will biuld graph before logits layer also without dropout, but now
          # what I use is pre logigts feature which is after dropout layer
          # TODO it should set num_classes default as None but slim return net just after global_pool, and for my code I used to use PreLogits 
          # after that, so here the hack is by default setting it to 1001, if you are using other pretrain model you might need to 
          # manually set num_classes like num_classes=5000(for openimage pretrain multilabel model) even if you just need the feature 
          # before fc layer(logits)  HACK! otherwise you might face problem loading model complaing wrong shape!
          if num_classes is None:
            num_classes = 1001 
          # TODO might modify to let scope be '' ?
          logging.info('pretrain image model num_classes:{}'.format(num_classes))
          net_fn = nets_factory.get_network_fn(net_name, num_classes=num_classes, is_training=is_image_model_training)
          logits, end_points = net_fn(image)

          # from nets import inception
          # with slim.arg_scope(inception.inception_resnet_v2_arg_scope()): 
          #   logits, end_points = inception.inception_resnet_v2(image, is_training=is_image_model_training, create_aux_logits=True)
          
          # for key in end_points:
          # print(key, end_points[key].shape)
          # print('end_points', end_points, file=sys.stderr)
          # if feature name is None will get final feature, other wise will get final attention feature
          # if feature in end point will directly get it other wise get from info[image_name][features_end_point]
          # TODO final end point for feature should be global_pool ? which is btter global_pool without dropout or PreLogits?
          # Try to finetune with using feature_name global_pool, or may be just use Logits (1001 dim as feature ?)

          print('feature_name', feature_name, file=sys.stderr)
          if feature_name and feature_name.lower() == 'logits':
            print('using slim image model logits', file=sys.stderr)
            image_feature = logits
            assert num_classes
          else:
            if (not feature_name) or (feature_name == 'final') or (feature_name.lower == 'none'):
              # None or empty feature name or final get final single feature
              print('image_model feature_name is None will get PreLogits or PreLogitsFlatten', file=sys.stderr)
              #print('end_point', end_points)
              if 'PreLogitsFlatten' in end_points:
                image_feature = end_points['PreLogitsFlatten']
              elif 'PreLogits' in end_points:
                net = end_points['PreLogits']
                image_feature = slim.flatten(net, scope="flatten")
              else:
                print('not found pre logits! get default final feature', file=sys.stderr)
                feature_name = info[name]['feature_end_point']
                print('image_model will get feature_name %s' % feature_name, file=sys.stderr)
                net = end_points[feature_name]
                image_feature = slim.flatten(net, scope="flatten")
            else:
              # get attention features
              if feature_name not in end_points:
                feature_name = info[name]['features_end_point']
              image_feature = end_points[feature_name]
              image_feature = slim.flatten(image_feature)

          print('image_feature:', image_feature, file=sys.stderr)
          #TODO check is it really ok? not finetune? seems still slow as im2txt it should be much faster then fintune.. FIXME?
          #TODO other method set not trainable, need to modify slim get_network_fn ?
          #if not trainable: #just for safe.. actuall slim.arg_scope with train_able=False works
          #  image_feature = tf.stop_gradient(image_feature)  
          if finetune_end_point: #None or ''
            logging.info('fintune image model from end point:{} {}'.format(finetune_end_point, end_points[finetune_end_point]))
            tf.stop_gradient(end_points[finetune_end_point])
          elif trainable:
            logging.info('fintune all image model layers')

          #--below is the same for inception v3
          # image_feature = melt.image.image_embedding.inception_v3(
          #   image_feature,
          #   trainable=trainable,
          #   is_training=is_training,
          #   reuse=reuse,
          #   scope=scope)

          #if not set this eval_loss = trainer.build_train_graph(eval_image_feature, eval_text, eval_neg_text) will fail
          #but still need to set reuse for melt.image.image_embedding.inception_v3... confused.., anyway now works..
          #with out reuse=True score = predictor.init_predict() will fail, resue_variables not work for it..
          #trainer create function once use it second time(same function) work here(with scope.reuse_variables)
          #predictor create another function, though seem same name same scope, but you need to set reuse=True again!
          #even if use tf.make_template still need this..
          #got it see hasky/jupter/scope.ipynb, because train then  eval, use same fn() call again in eval scope.reuse_varaibbles() will in effect
          #escape_fn3 = create_escape_construct_fn('XXX')
          #escape_fn3()
          #escape_fn3() #ok becasue scope.reuse_variables() here
          #but for predictor escape_fn3 = create_escape_construct_fn('XXX') you call it again, then escape_fn3() will fail need reuse

          scope.reuse_variables() #this is fine make function() '' scope resue, set True, but if not use function with ... will fail also
          return image_feature

    return construct_fn

#TODO... not in affect... Still not very clear with make_template and create_scope_now_ (google/seq2seq set this True, default is False)
# it is like above useing scope.reuse_variables() right after.. I suppose

"""
#encode_fn = SomeEncoderModule(...)

## New variables are created in this call.
#output1 = encode_fn(input1)

## No new variables are created here. The variables from the above call are re-used.
## Note how this is different from normal TensorFlow where you would need to use variable scopes.
#output2 = encode_fn(input2)

## Because this is a new instance a second set of variables is created.
#encode_fn2 = SomeEncoderModule(...)
#output3 = encode_fn2(input3)
"""

#this will be safe as build graph as root scope but has problem always building using image_processing?
#from textsum seems not affected, will not create graph if you not use this function
#_image2feature_fn = create_image2feature_fn()
#make this function to be resued/share without manully set scope reuse
#image2feature_fn = tf.make_template('image2feature', _image2feature_fn, create_scope_now_=True)
#image2feature_fn = tf.make_template('image2feature', _image2feature_fn, create_scope_now_=False)
#image2feature_fn =  create_image2feature_fn()
