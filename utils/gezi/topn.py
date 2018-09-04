#!/usr/bin/env python
# ==============================================================================
#          \file   topn.py
#        \author   chenghuige  
#          \date   2017-03-12 16:55:52.257425
#   \Description  
# ==============================================================================

"""
copy from 
tensorflow\models\im2txt\im2txt\inference_utils\caption_generator.py
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import heapq

class TopN(object):
  """Maintains the top n elements of an incrementally provided set."""

  def __init__(self, n, reverse=True):
    self._n = n
    self._data = []
    self.reverse = reverse

  def size(self):
    assert self._data is not None
    return len(self._data)

  def push(self, x):
    """Pushes a new element."""
    assert self._data is not None
    if len(self._data) < self._n:
      heapq.heappush(self._data, x)
    else:
      heapq.heappushpop(self._data, x)

  def extract(self, sort=False):
    """Extracts all elements from the TopN. This is a destructive operation.

    The only method that can be called immediately after extract() is reset().

    print(id, logprob)
    Args:
      sort: Whether to return the elements in descending sorted order.

    Returns:
      A list of data; the top n elements provided to the set.
    """
    assert self._data is not None
    data = self._data
    self._data = None
    if sort:
      data.sort(reverse=self.reverse)
    return data

  def reset(self):
    """Returns the TopN to an empty state."""
    self._data = []
