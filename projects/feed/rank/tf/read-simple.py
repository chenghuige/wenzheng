#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   read-simple.py
#        \author   chenghuige  
#          \date   2019-07-30 22:29:54.489845
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import tensorflow as tf
#tf.enable_eager_execution()


#创建一个Dataset对象
dataset = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6,7,8,9, 10])

'''合成批次'''
dataset=dataset.batch(3)

#创建一个迭代器
iterator = dataset.make_one_shot_iterator()

#get_next()函数可以帮助我们从迭代器中获取元素
element = iterator.get_next()

#遍历迭代器，获取所有元素
with tf.Session() as sess:
   for i in range(9):
       print(sess.run(element)) 
