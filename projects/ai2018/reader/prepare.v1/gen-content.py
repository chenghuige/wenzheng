#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-content.py
#        \author   chenghuige  
#          \date   2018-09-11 10:37:16.658519
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

infile = './mount/data/ai2018/reader/train.json'

import json 

for line in open(infile):
  m = json.loads(line.rstrip())
  print(m['query'])
  print(m['passage'])

