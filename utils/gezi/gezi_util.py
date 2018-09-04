from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time 
import sys, os
from collections import defaultdict

def gprint(convert2utf8, *content):
  if convert2utf8:
    print(' '.join([toutf8(str(x)) for x in content]))
  else:
    print(' '.join([str(x) for x in content]))

def uprint(*content):
  print(' '.join([toutf8(str(x)) for x in content]))

def toutf8(content):
  return content.decode('gbk', 'ignore').encode('utf8', 'ignore')

def togbk(content):
  return content.decode('utf8', 'ignore').encode('gbk', 'ignore')

from datetime import datetime
def now_time():
  return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def get_timestr(stamp): 
  ttime = stamp
  try:
    ttime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stamp))
  except Exception:
    pass
  return ttime

def get_datestr(stamp):
  ttime = stamp
  try:
    ttime = time.strftime('%Y-%m-%d 00:00:00', time.localtime(stamp))
  except Exception:
    pass
  return ttime

def pretty_print(df):
  from StringIO import StringIO
  import prettytable    

  output = StringIO()
  df.to_csv(output, index = False, sep ='\t')
  output.seek(0)
  pt = prettytable.from_csv(output)
  print(pt)

def print_list(l, sep='|'):
  for item in l:
    print(item, sep, end='')
  print()

def get_words(l, ngram, sep = '\x01'):
  li = []
  for i in range(len(l)):
    for j in range(ngram):
      if (i + j >= len(l)):
        continue 
      li.append(sep.join(l[i : i + j + 1]))
  return li 

def get_ngram_words(l, ngram, sep = '\x01'):
  li = []
  for i in range(len(l)):
    for j in range(ngram):
      if (i + j >= len(l)):
        continue 
      li.append(sep.join(l[i : i + j + 1]))
  return li 

#@TODO generalized skip n now only skip-n bigram
#skip 1 2 3 4
def get_skipn_bigram(l, n, sep = '\x01'):
  li = []
  l2 = ['0', '0']
  for i in range(len(l) - 1 - n):
    l2[0] = l[i]
    l2[1] = l[i + 1 + n]
    li.append(sep.join(l2))
  return li 

def get_skipn_bigram(l, li, n, sep = '\x01'):
  l2 = ['0', '0']
  for i in range(len(l) - 1 - n):
    l2[0] = l[i]
    l2[1] = l[i + 1 + n]
    li.append(sep.join(l2))

def get_skip_bigram(l, li, n, sep = '\x01'):
  for skip in range(1, n):
    get_skipn_bigram(l, li, skip, sep)

def h2o(x):
  if isinstance(x, dict):
    return type('jo', (), {k: h2o(v) for k, v in x.iteritems()})
  elif isinstance(x, list):
    return [h2o(item) for item in x]
  else:
    return x

def h2o2(x):
  if isinstance(x, dict):
    return type('jo', (), {k: h2o2(v) for k, v in x.iteritems()})
  elif isinstance(x, list):
    return type('jo', (), {"value" + str(idx): h2o2(val) for idx, val in enumerate(x)})
  else:
    return x

def json2obj(s):
  import json
  a = h2o(json.loads(s))
  if hasattr(a, 'data'):
    return a.data
  return a

def json2obj2(s):
  import json
  a = h2o2(json.loads(s))
  if hasattr(a, 'data'):
    return a.data
  return a

def jsonfile2obj(path):
  return json2obj(open(path).read().decode('gbk', 'ignore'))

def jsonfile2obj2(path):
  return json2obj2(open(path).read().decode('gbk','ignore'))

def xmlfile2obj(path):
  import xmltodict
  doc = xmltodict.parse(open(path), process_namespaces=True)
  a = h2o2(doc)
  if hasattr(a, 'cereal'):
    return a.cereal.data
  return a

def xmlfile2obj2(path):
  import xmltodict
  doc = xmltodict.parse(open(path), process_namespaces=True)
  a = h2o2(doc)
  if hasattr(a, 'cereal'):
    return a.cereal.data
  return a

#from matplotlib.pylab import show
#def show(s = 'temp.png'):
#	import matplotlib.pylab as pl
#	pl.savefig(s)

#from matplotlib.pylab import clf

def dict2map(dict_, map_):
  for key,value in dict_.items():
    map_[key] = value 

def map2dict(map_):
  dict_ = {}
  for item in map_:
    dict_[item.key] = item.data 
  return dict_

def list2vec(list_, vec_):
  for item in list_:
    vec_.append(item)

def list2vector(list_, vec_):
  for item in list_:
    vec_.push_back(item)

def vec2list(vec_):
  list_ = []
  for item in vec_:
    list_.append(item)
  return list_

def vector2list(vec_):
  list_ = []
  for i in xrange(vec_.size()):
    list_.append(vec_[i])
  return list_

def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.
    
def get_num_lines(file):
  count = 0
  for count, line in enumerate(open(file)):
    pass
  return count + 1
