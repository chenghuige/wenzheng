#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   mpi.py
#        \author   chenghuige  
#          \date   2019-07-30 08:29:49.196748
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

# mpi_helloworld.py

from mpi4py import MPI


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
node_name = MPI.Get_processor_name() # get the name of the node

print('Hello world from process %d at %s.' % (rank, node_name))
