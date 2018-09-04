#! /usr/bin/env python
#coding=gbk
############################################
# [Author]: jianglianxin, jianglianxin
# [Date]:	2013/01/15
############################################
import pydictmatch
from sets import Set
import sys

class Pydict :
  def __init__(self,path_dm) :
    self.dm_dict_handle = pydictmatch.dm_binarydict_load(path_dm)
    self.dm_pack_handle = pydictmatch.dm_pack_create()
    if (not self.dm_dict_handle) or (not self.dm_pack_handle):
      print >>sys.stderr, "load pydict fail"
      sys.exit(-1)
  
  def search(self,query,option):
    ##option 1:match loggest 0: match all
    matchList = []
    if option != 0 and option != 1:
      return matchList
    matchList = pydictmatch.dm_search(self.dm_dict_handle, self.dm_pack_handle,query,option)
    return matchList
  
  def close(self) :
    pydictmatch.dm_pack_del(self.dm_pack_handle)
    pydictmatch.dm_dict_del(self.dm_dict_handle)

