#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   bcast.py
#        \author   chenghuige  
#          \date   2019-07-31 15:17:38.283445
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import horovod.tensorflow as hvd
import mpi4py
from mpi4py import MPI
comm = MPI.COMM_WORLD

mpi4py.rc.initialize = False
hvd.init()
assert hvd.mpi_threads_supported()
assert hvd.size() == comm.Get_size()

rank = comm.Get_rank()

if rank == 0:
    s = 'abcdef'
    data = {'key1' : [7, 2.72, 2+3j],
            'key2' : ( 'abc', 'xyz')}
    print('before broadcasting: process %d has %s' % (rank, data), s)
else:
    s = None
    data = None
    print('before broadcasting: process %d has %s' % (rank, data), s)

data = comm.bcast(data, root=0)
s = comm.bcast(s, root=0)
print('after broadcasting: process %d has %s' % (rank, data), s)

  
