#!/usr/bin/env python
# ==============================================================================
#          \file   run.py
#        \author   chenghuige  
#          \date   2017-11-04 23:23:38.351100
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

jobs = open('./commands.py').readlines()
jobs = [x.strip() for x in jobs if x.strip() and not x.startswith('#')]

queue = '3-8'

if len(sys.argv) > 1:
  queue = sys.argv[1]

prefix=''

for job in jobs:
  if job.strip().startswith('#'):
    continue
  num_gpus = 1
  l = job.split('|')
  if len(l) == 2:
    num_gpus = int(l[-1].strip())
  elif len(l) == 3:
    queue = l[-2].strip()

  command = 'cat run-head.txt > run.sh'
  os.system(command)
  
  if job.strip().startswith('['):
    prefix = job.strip()[1:-1]
    continue

  job = l[0].strip()
  with open('./run.sh', 'a') as out: 
    if not prefix:
      print(job, file=out)
    else:
      print('%s%s' % (prefix, job))
      print('%s%s' % (prefix, job), file=out)
  print('----', job, file=sys.stderr)
  if '/' in job:
    name = '.'.join(job.split()[1].split('/')[-1].split('.')[:-1])
  else:
    name = '.'.join(job.split('.')[:-1])
  if prefix:
    #name = name + '_' + prefix.split()[-1].replace('=', '_')
    name = name + '_' + '_'.join([x.replace(';', '').split('=')[-1] for x in prefix.split() if '=' in x])
  if num_gpus > 1:
    name = '%s.%d' % (name, num_gpus)
  command = 'sh submit.sh %s %s %d' % (name, queue, num_gpus)
  print(command, file=sys.stderr)
  os.system(command)
  
  
