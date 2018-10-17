#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   logging.py
#        \author   chenghuige  
#          \date   2016-09-24 09:25:56.796006
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

import sys

import logging
import logging.handlers

import gezi

_logger = logging.getLogger('melt')

_logger2 = logging.getLogger('melt2')
   
#_handler = logging.StreamHandler()
#_handler.setLevel(logging.INFO)
#_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
#_logger.addHandler(_handler)
#_logger.setLevel(logging.INFO)

log = _logger.log
#debug = _logger.debug
#error = _logger.error
#fatal = _logger.fatal
#info = _logger.info
#warn = _logger.warn
#warning = _logger.warning  

#info2 = _logger2.info

def info(*args):
  _logger.info(' '.join("{}".format(a) for a in args))

def info2(*args):
  _logger2.info(' '.join("{}".format(a) for a in args))

def fatal(*args):
  _logger.fatal(' '.join("{}".format(a) for a in args))

def error(*args):
  _logger.error(' '.join("{}".format(a) for a in args))

def debug(*args):
  _logger.debug(' '.join("{}".format(a) for a in args))

def warn(*args):
  _logger.warn(' '.join("{}".format(a) for a in args))

def warning(*args):
  _logger.warning('WARNING: %s' % (' '.join("{}".format(a) for a in args)))

from datetime import timedelta
import time

class ElapsedFormatter():
  def __init__(self):
    self.start_time = time.time()
  
  def format(self, record):
    elapsed_seconds = record.created - self.start_time
    #using timedelta here for convenient default formatting
    elapsed = timedelta(seconds = elapsed_seconds)
    return "{} {} {}".format(gezi.now_time(), str(elapsed)[:-7], record.getMessage())

_logging_file = None
_logging_file2 = None

def _get_handler(file, formatter, split=True, split_bytime=False, mode = 'a', level=logging.INFO):
  #setting below will set root logger write to _logging_file
  #logging.basicConfig(filename=_logging_file, level=level, format=None)
  #logging.basicConfig(filename=_logging_file, level=level)
  #save one per 1024k/4, save at most 10G 1024
  if split:
    if not split_bytime:
      file_handler = logging.handlers.RotatingFileHandler(file, mode=mode, maxBytes=1024*1024/4, backupCount=10240*4)
    else:
      file_handler = logging.handlers.TimedRotatingFileHandler(file, when='H', interval=1, backupCount=1024)
      file_handler.suffix = "%Y%m%d-%H%M"
  else:
    file_handler = logging.FileHandler(_logging_file, mode=mode)  
  file_handler.setLevel(level)  
  file_handler.setFormatter(formatter)
  return file_handler

def set_logging_path(path, file='log.html', logtostderr=True, logtofile=True, split=True, split_bytime=False, level=logging.INFO, mode='a'):
  global _logger, _logging_file
  if _logging_file is None:
    if not path:
      path = '/tmp/'
    _logging_file = '%s/%s'%(path, file)
    _logging_file2 = '%s/log.txt'%path
    #formatter = logging.Formatter("%(asctime)s %(message)s",
    #                          "%Y-%m-%d %H:%M:%S")
    formatter = ElapsedFormatter()
    if logtofile:
      gezi.try_mkdir(path)
      file_handler = _get_handler(_logging_file, formatter, split, split_bytime, mode, level)
      file_handler2 = _get_handler(_logging_file2, formatter, split, True, mode, level)
      #file_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
      _logger.addHandler(file_handler)
      _logger2.addHandler(file_handler2)
  
    if logtostderr:
      handler = logging.StreamHandler()
      handler.setLevel(logging.INFO)
      handler.setFormatter(formatter)
      #handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
      _logger.addHandler(handler)
      _logger2.addHandler(handler)
      
      # some how new tf cause to logg twice.. 
      # https://stackoverflow.com/questions/19561058/duplicate-output-in-simple-python-logging-configuration
      _logger.propagate = False 
      _logger2.propagate = False
  
    _logger.setLevel(level)
    _logger2.setLevel(level)

def init(file='log.html', mode='a', logtostderr=False, logtofile=True, path='./', split=False, split_bytime=False, level=logging.INFO):
  logging.basicConfig(level=logging.INFO, stream=sys.stdout)
  set_logging_path(path=path, mode=mode, file=file, logtostderr=logtostderr, logtofile=logtofile, split=split, split_bytime=split_bytime, level=level)

def vlog(level, msg, *args, **kwargs):
  _logger.log(level, msg, *args, **kwargs)

def get_verbosity():
  """Return how much logging output will be produced."""
  return _logger.getEffectiveLevel()

def set_verbosity(verbosity):
  """Sets the threshold for what messages will be logged."""
  _logger.setLevel(verbosity)

def get_logging_file():
  return _logging_file