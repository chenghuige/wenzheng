#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   libgezi_util.py
#        \author   chenghuige  
#          \date   2016-08-25 21:08:23.051951
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import gezi.nowarning

"""
@TODO gezi should remove boost.python dependence
now has libgezi_util and segment  depend on boost.python
"""
#@FIXME this might casue double free at the end, conflict with numpy in virutal env
import gezi

if gezi.encoding == 'gbk' or gezi.env_has('BAIDU_SEG'):
  import libgezi # must include this not sure why..
  import libstring_util as su
  def get_single_cns(text):
    return su.to_cnvec(su.extract_chinese(text)) 
  
  def is_single_cn(word):
    word = word.decode('gbk', 'ignore')
    return u'\u4e00' <= word <= u'\u9fff'

  def get_single_chars(text):
    l = [x.encode('gbk') for x in text.decode('gbk', 'ignore')]
    return [x.strip() for x in l if x.strip()]

else:
  def get_single_cns(text):
    l = []
    pre_is_cn = False 
    if six.PY2:
      text = text.decode('utf-8', 'ignore')
    for word in text:
      if u'\u4e00' <= word <= u'\u9fff':
        pre_is_cn = True
        if l:
          l.append(' ')
      else:
        if pre_is_cn:
          l.append(' ')
          pre_is_cn = False
      if pre_is_cn:
        l.append(word)
    text = ''.join(l) 
    if six.PY2:
      text = text.encode('utf-8')
    l = text.split()
    return [x.strip() for x in l if x.strip()]    

  def is_cn(word):
    if six.PY2:
      word = word.decode('utf-8', 'ignore')
    return u'\u4e00' <= word <= u'\u9fff'

  def is_single_cn(word):
    return len(get_single_cns(word)) == 1

  def get_single_chars(text):
    if six.PY2:
      l = [x.encode('utf8') for x in text.decode('utf8', 'ignore')]
    else:
      l = [x for x in text]
    return [x.strip() for x in l if x.strip()]
