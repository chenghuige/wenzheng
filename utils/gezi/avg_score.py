#!/usr/bin/env python
# ==============================================================================
#          \file   avg_loss.py
#        \author   chenghuige  
#          \date   2016-08-17 12:01:41.003969
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class AvgScore():
  """
  Calcuatlate avg scores, input can be single value or list
  Not using numpy, so may be much slower then numpy version, now mainly used for tensorflow testers
  """
  def __init__(self):
    self.reset()
  
  def reset(self):
    self.num_steps = 0
    self.total_score = 0.
    self.total_scores = None

  def add(self, score):
    if isinstance(score, list):
      if self.total_scores is None:
        self.total_scores = score
      else:
        self.total_scores =[sum(x) for x in zip(self.total_scores, score)]
    else: 
      self.total_score += score
    self.num_steps += 1

  #@property
  def avg_score(self):
    if self.total_scores is None:
      avg_score_ = self.total_score / self.num_steps 
    else:
      avg_score_ = [x / self.num_steps for x in self.total_scores]
    self.reset()
    return avg_score_
    
  
