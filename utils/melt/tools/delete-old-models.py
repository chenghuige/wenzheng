#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   delete-old-models.py
#        \author   chenghuige  
#          \date   2016-10-10 21:58:26.617674
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
bin model_dir num_to_keep_of_latest_models
"""

import sys, os, glob

def list_models(model_dir, time_descending=True):
  """
  list all models in model_dir
  """
  files = [file for file in glob.glob('%s/model.ckpt-*'%(model_dir)) if not (file.endswith('.meta') or file.endswith('.index'))]
  files.sort(key=lambda x: os.path.getmtime(x), reverse=time_descending)
  return files 

model_dir = sys.argv[1]
models = list_models(model_dir)
print('total models now:', len(models))
os.system('du -h %s'%model_dir)

if len(sys.argv) > 2:
  num_keeps = int(sys.argv[2])
  print('The models to keep are', models[:num_keeps])
  print('You want to keep only the latest %d  models, so delete %d old models'%(num_keeps, len(models) - num_keeps))
  
  is_first = True
  for model in models[num_keeps:]:
    if is_first:
      if num_keeps <  5:
        print('only keep %d models?'%num_keeps)
      print('delete model older then %s ?'%model)
      #ok = raw_input("y?: ")
      #if ok != 'y' and ok != 'yes':
      #	break
      is_first = False
    print('delete model:', model)
    os.remove(model)
    try:
      l = model.split('.')
      if not l[-1].startswith('ckpt-'):
        l = l[:-1]
      model = '.'.join(l)
      os.remove('%s.meta'%model)
      os.remove('%s.index'%model)
    except Exception:
      pass

models = list_models(model_dir)
print('models leave after:', models, 'num models now', len(models))
os.system('du -h %s'%model_dir)
