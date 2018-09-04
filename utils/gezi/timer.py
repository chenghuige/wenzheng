#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   timer.py
#        \author   chenghuige  
#          \date   2016-08-15 16:32:21.015897
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
@TODO may be

with gezi.Timer('abc') as timer:
  ....
"""
import sys, time
class Timer():
  def __init__(self, info='', print_before=True):
    self.start_time = time.time()
    if info and print_before:
      print('%s start'%info, file=sys.stderr)
    self.info = info

  def elapsed(self):
    end_time = time.time()
    duration = end_time - self.start_time
    self.start_time = end_time 
    return duration  

  def elapsed_ms(self):
    return self.elapsed() * 1000
  
  #ipython not allow this?.. FIXME
  def print(self):
    if self.info:
      print('{} duration: {}'.format(self.info, self.elapsed()), file=sys.stderr)
    else:
      print(self.elapsed(), file=sys.stderr)

  def print_elapsed(self):
    if self.info:
      print('{} duration: {}'.format(self.info, self.elapsed()), file=sys.stderr)
    else:
      print(self.elapsed(), file=sys.stderr)