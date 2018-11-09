#!/usr/bin/env python
# ==============================================================================
#          \file   losses.py
#        \author   chenghuige  
#          \date   2017-08-07 13:11:10.672206
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import sys, os

import tensorflow as tf 
import melt

def reduce_loss(loss_matrix, combiner='mean'):
  if combiner == 'mean':
    return tf.reduce_mean(loss_matrix)
  elif combiner == 'sum':
    return tf.reduce_sum(loss_matrix)
  elif combiner == 'max':
    return tf.reduce_mean(tf.reduce_max(loss_matrix, 1))

#https://stackoverflow.com/questions/37479119/doing-pairwise-distance-computation-with-tensorflow
#https://hackernoon.com/one-shot-learning-with-siamese-networks-in-pytorch-8ddaab10340e
#https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
#http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

#there are different versions, one is not to use sqrt.. just square
#input score not l2 distance withou sqrt
def contrastive(pos_score, neg_scores, margin=1.0, use_square=True, combiner='mean', name=None,):
  #relu same like hinge.. tf.max..
  #neg_scores = tf.nn.relu(margin - neg_scores)
  neg_scores = tf.nn.relu(margin - tf.sqrt(neg_scores))
  if use_square:  
    neg_scores = tf.square(neg_scores)
  else:
    pos_score = tf.sqrt(pos_score)
  
  scores = tf.concat([pos_score, neg_scores], 1)
  loss = reduce_loss(scores, combiner) * 0.5
  return loss

def triplet(pos_score, neg_scores, margin=1.0, combiner='mean', name=None,):
  #margin small then loss turn to zero quick, margin big better diff hard images but hard to converge
  #if pos_score(dist) is smaller then neg_score more then margin then loss is zero
  scores = tf.nn.relu(margin - (neg_scores - pos_score))
  return reduce_loss(scores, combiner)

#this is cross entorpy for cosine same... scores must be -1 <-> 1 TODO
def cross_entropy(scores, combiner='mean', name=None):
  with tf.name_scope(name, 'cross_entropy_loss', [scores]):
    batch_size = scores.get_shape()[0]
    num_negs = scores.get_shape()[1] - 1
    targets = tf.concat([tf.ones([batch_size, 1], tf.float32), tf.zeros([batch_size, num_negs], tf.float32)], 1)
    #http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/ 
    #I think for binary is same for sigmoid or softmax
    
    #logits = tf.sigmoid(scores)
    
    #logits = (1. + scores) / 2.

    logits = scores

    loss_matrix = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets)
    #loss_matrix = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    loss = reduce_loss(loss_matrix, combiner)
    return loss

#---------below pairwise
def hinge(pos_score, neg_score, margin=0.1, combiner='mean', name=None):
  with tf.name_scope(name, 'hinge_loss', [pos_score, neg_score]):
    ##so if set 0.5 margin , will begin loss from 0.5 since pos and neg sore most the same at begin
    #loss_matrix = tf.maximum(0., margin - (pos_score - neg_score))
    loss_matrix = tf.nn.relu(margin - (pos_score - neg_score))
    loss = reduce_loss(loss_matrix, combiner)
    return loss

def pairwise_cross(pos_score, neg_score, combiner='mean', name=None):
  with tf.name_scope(name, 'hinge_cross_loss', [pos_score, neg_score]):
    score = pos_score - neg_score
    #logits = tf.sigmoid(score)
    logits = score
    targets = tf.ones_like(neg_score, tf.float32)
    loss_matrix = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets)
    loss = reduce_loss(loss_matrix, combiner)
    return loss

def pairwise_exp(pos_score, neg_score, theta=1.,  combiner='mean', name=None):
  with tf.name_scope(name, 'hinge_exp_loss', [pos_score, neg_score]):
    score = pos_score - neg_score
    loss = tf.log(1 + tf.reduce_sum(tf.exp(-theta * score)))
    #loss = tf.log(1. + tf.exp(-theta * score))
    #loss = reduce_loss(loss, combiner)
    return loss

# https://github.com/tflearn/tflearn/blob/184d753f8fe6ab82a5033f6cbef8edc91b40ca8c/tflearn/objectives.py
def roc_auc_score(y_pred, y_true):
    """ ROC AUC Score.
    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.
    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
    Measures overall performance for a full range of threshold levels.
    Arguments:
        y_pred: `Tensor`. Predicted values.
        y_true: `Tensor` . Targets (labels), a probability distribution.
    """
    with tf.name_scope("RocAucScore"):

        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)

        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.2
        p     = 3

        difference = tf.zeros_like(pos * neg) + pos - neg - gamma

        masked = tf.boolean_mask(difference, difference < 0.0)

        return tf.reduce_sum(tf.pow(-masked, p))
        #return tf.pow(-masked, p)



#https://github.com/thinline72/toxic/blob/master/skolbachev/toxic/losses.py

def getClassWeights(Y, mu=0.5):
    return np.array([w for w in np.log(mu*Y.shape[0]/Y.sum(axis=0))])

# from keras import backend as K
# def focal_loss(y_true, y_pred, alpha, gamma=0.5):
#     alpha = K.variable(alpha)
#     pt = K.abs(1. - y_true - y_pred)
#     pt = K.clip(pt, K.epsilon(), 1. - K.epsilon())
#     return K.mean(-alpha * K.pow(1. - pt, gamma) * K.log(pt), axis=-1)

def u_statistic_loss(y_true, y_pred, gamma=0.2, p=3.0):
    """ U statistic loss
    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.
    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
    Measures overall performance for a full range of threshold levels.
    Arguments:
        y_pred: `Tensor`. Predicted values.
        y_true: `Tensor` . Targets (labels), a probability distribution.
    """
    with tf.name_scope("u_statistic_loss"):
        
        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))
        
        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)
        
        difference = tf.zeros_like(pos * neg) + pos - neg - gamma
        masked = tf.boolean_mask(difference, difference < 0.0)
        return tf.reduce_sum(tf.pow(-masked, p))

def SoftAUC_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    parts = tf.dynamic_partition(y_pred, y_true, 2)
    y_pos = parts[1]
    y_neg = parts[0]
    y_pos = tf.expand_dims(y_pos, 0)
    y_neg = tf.expand_dims(y_neg, -1)
    return tf.reduce_mean(tf.nn.sigmoid(y_neg - y_pos))

def SVMrank_loss(y_true, y_pred):
    margin = 1.0
    y_true = tf.cast(y_true, tf.int32)
    parts = tf.dynamic_partition(y_pred, y_true, 2)
    y_pos = parts[1]
    y_neg = parts[0]
    y_pos = tf.expand_dims(y_pos, 0)
    y_neg = tf.expand_dims(y_neg, -1)
    return tf.reduce_mean(tf.nn.relu(margin - y_neg - y_pos))


###########experimental losses##############

def exp_loss(y_true, y_pred):
    loss = u_statistic_loss(y_true,y_pred) + SoftAUC_loss(y_true, y_pred)
    return loss

def art_loss(y_true, y_pred):
    loss = u_statistic_loss(y_true,y_pred) + SVMrank_loss(y_true, y_pred)
    return loss
    
def roc_auc_scores(y_pred, y_true):
  num_classes = melt.get_shape(y_pred, -1)
  y_preds = tf.split(y_pred, num_classes, axis=1)
  y_trues = tf.split(y_true, num_classes, axis=1)
  losses = []
  for y_pred, y_true in zip(y_preds, y_trues):
    losses.append(roc_auc_score(y_pred, y_true)) 
    #losses.append(art_loss(y_pred, y_true))
  return tf.stack(losses)


# https://github.com/ailias/Focal-Loss-implement-on-Tensorflow/blob/master/focal_loss.py

from tensorflow.python.ops import array_ops
def focal_loss(target_tensor, prediction_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    
    #target_tensor = tf.to_float(target_tensor)
    target_tensor = tf.one_hot(target_tensor, melt.get_shape(prediction_tensor, -1))

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    #return tf.reduce_sum(per_entry_cross_ent)
    return tf.reduce_mean(per_entry_cross_ent)

def earth_mover_loss(y_true, y_pred):	
  cdf_ytrue = tf.cumsum(y_true, axis=-1)
  cdf_ypred = tf.cumsum(y_pred, axis=-1)
  samplewise_emd = tf.sqrt(tf.reduce_mean(tf.square(tf.abs(cdf_ytrue - cdf_ypred)), axis=-1))
  #return samplewise_emd
  return tf.reduce_mean(samplewise_emd)

def bilm_loss(model, x, y, training=False):
  zero_col = tf.expand_dims(tf.zeros_like(y[:,0]), 1)
  fw_y = tf.concat([y[:, 1:], zero_col], 1)
  bw_y = tf.concat([zero_col, y[:, 0:-1]], 1)

  fw_mask = fw_y > 0
  bw_mask = bw_y > 0  

  ## Well intersting if you just use model.encode() then you will get text_encoder/ as namespace
  ## if you use model() then you will get like rnet/text_encoder or bi_language_model
  ## TODO FIXME so by default should use model.encode() that but now I want to use rent pretrain only just set this..
  # so how to better handle tf pretrain and finetune, bert code need to learn later! maybe that's best way though still a bit complex
  #y_ = model.encode(x, training=training)
  y_ = model(x, training=training)
  
  fw_y_, bw_y_ = tf.split(y_, 2, axis=-1)

  fw_y_ = model.encode.hidden2tag(fw_y_)
  bw_y_ = model.encode.hidden2tag(bw_y_)

  fw_weight = tf.to_float(fw_mask)
  bw_weight = tf.to_float(bw_mask)

  fw_loss = melt.seq2seq.sequence_loss_by_example(fw_y_, fw_y, fw_weight)
  bw_loss = melt.seq2seq.sequence_loss_by_example(bw_y_, bw_y, bw_weight)

  fw_loss = tf.reduce_mean(fw_loss)
  bw_loss = tf.reduce_mean(bw_loss)

  # TODO FIXME melt should check return loss is scalar
  loss = (fw_loss + bw_loss) / 2.

  return loss
  
  
