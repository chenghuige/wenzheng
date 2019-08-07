#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   bigdata_util.py
#        \author   chenghuige  
#          \date   2016-08-26 17:43:05.542995
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#@TODO glob and tf.GFile.Glob diff?
import sys
import glob as local_glob

try:
  import pyhdfs
except Exception:
  pass

HDFS_START = 'hdfs://'

#@TODO config ?
NODE_ADDRESS = ''
NODE_PORT = 54310
USER_NAME = ''
USE_PASSWORD = ''

START_DIR='/app/'

handle = None 

def init():
  global handle
  if handle is None:
    handle = pyhdfs.hdfsConnectAsUser(NODE_ADDRESS, NODE_PORT, USER_NAME, USE_PASSWORD)
  return handle 

def get_handle():
  init()
  return handle

def fullpath(path):
  return '{}{}:{}/{}'.format(HDFS_START, NODE_ADDRESS, NODE_PORT, path)

def glob(file_pattern):
  """
  @depreciated
  Now I only find pyhdfs.hdfsListDirectory, @TODO support hdfsGlob 
  """
  if file_pattern.startswith(HDFS_START):
    handle = get_handle()
    # hdfs://
    return pyhdfs.hdfsListDirectory(handle, file_pattern)
  elif file_pattern.startswith(START_DIR):
    handle = get_handle()
    # /app/tuku/..
    result, num = pyhdfs.hdfsListDirectory(handle, fullpath(file_pattern))
    return [item.mName for item in [pyhdfs.hdfsFileInfo_getitem(result, i) for i in xrange(num)] if item.mSize > 0]
  return local_glob.glob(file_pattern)

def hdfs_listdir(dir):
  #now only support listdir not glob
  handle = get_handle()
  result, num = pyhdfs.hdfsListDirectory(handle, fullpath(dir))
  final_result = [item.mName for item in [pyhdfs.hdfsFileInfo_getitem(result, i) for i in xrange(num)] if item.mSize > 0]
  print('ori result num:{}, final result num:{}'.format(num, len(final_result)), file=sys.stderr)
  return final_result

import gezi
def list_files(input):
  """
  @TODO support hdfsGlob 
  """
  if not input:
    return []
  local_files = gezi.list_files(input)
  if local_files:
    return local_files
  if not input.startswith(START_DIR):
    return []
  #now only support listdir not glob
  return hdfs_listdir(input)

def is_remote_path(path):
  return path.startswith(START_DIR)
  
