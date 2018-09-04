#!/usr/bin/env python
# ==============================================================================
#          \file   prepare-finetune.py
#        \author   chenghuige  
#          \date   2017-09-29 07:15:01.808002
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import melt  

model_dir = sys.argv[1]

model_dir, model_path = melt.get_model_dir_and_path(model_dir)

new_model_dir = sys.argv[2]
command = 'mkdir -p %s'%new_model_dir 
print(command, file=sys.stderr)
os.system(command)

if os.path.exists(new_model_dir):
  command = 'rm -rf %s/*'%new_model_dir 
  print(command, file=sys.stderr)
  os.system(command)

command = 'cp %s* %s'%(model_path, new_model_dir)
print(command, file=sys.stderr)
os.system(command)

checkpoint = os.path.join(new_model_dir, 'checkpoint')
model_name = os.path.basename(model_path)

checkpoint_info = 'model_checkpoint_path: "./%s"\n'%model_name 

print('write checkpoint_info to %s'%checkpoint, file=sys.stderr)
with open(checkpoint, 'w') as f:
  f.write(checkpoint_info)

old_checkpoint = os.path.join(model_dir, 'checkpoint')
from_checkpoint = os.path.join(new_model_dir, 'checkpoint.from')
command = 'cp %s %s'%(old_checkpoint, from_checkpoint)
print(command, file=sys.stderr)
os.system(command)
