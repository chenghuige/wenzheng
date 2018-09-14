#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   check.py
#        \author   chenghuige  
#          \date   2018-09-10 01:35:31.982045
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import json

infos = {}

for line in open(sys.argv[1]):
  m = json.loads(line.rstrip('\n'))
  query_id = str(m['query_id'])
  alternatives = m['alternatives']
  infos[query_id] = [x.strip() for x in alternatives.split('|')]
  if len(infos[query_id]) != 3:
    print(line)

for line in open(sys.argv[2]):
  query_id, predict = line.rstrip('\n').split('\t')
  if predict not in infos[query_id]:
    print(query_id, predict)
  
