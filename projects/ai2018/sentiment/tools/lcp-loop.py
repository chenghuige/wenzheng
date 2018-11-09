#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   lcp.py
#        \author   chenghuige  
#          \date   2018-05-23 13:06:34.938478
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

import glob 
import time

input = sys.argv[1]

base = os.path.basename(input)
os.system('mkdir -p %s' % base)

while True:
  for item in glob.glob(input + '/*'):
    if os.path.isdir(item):
      print(item)
      folder = os.path.basename(item)
      command = 'mkdir -p %s/%s' %(base, folder) 
      print(command)
      os.system(command)
      for item2 in glob.glob('%s/*' % item):
        if os.path.isdir(item2):
          folder2 = os.path.basename(item2)
          command = 'mkdir -p %s/%s/%s' % (base, folder, folder2)
          print(command)
          os.system(command)
          command = 'mkdir -p %s/%s/%s/epoch' %(base, folder, folder2)
          os.system(command)
          command = 'mkdir -p %s/%s/%s/ckpt' %(base, folder, folder2)
          os.system(command)
          command = 'mkdir -p %s/%s/%s/ckpt2' %(base, folder, folder2)
          os.system(command) 
          os.system('rsync --progress -avz %s/epoch/*valid* %s/%s/%s/epoch' % (item2, base, folder, folder2))
          os.system('rsync --progress -avz %s/epoch/*infer* %s/%s/%s/epoch' % (item2, base, folder, folder2))
          os.system('rsync --progress -avz %s/ckpt/*valid* %s/%s/%s/ckpt' % (item2, base, folder, folder2))
          os.system('rsync --progress -avz %s/ckpt/*infer* %s/%s/%s/ckpt' % (item2, base, folder, folder2))
  time.sleep(10)
  
