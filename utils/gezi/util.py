#!/usr/bin/env python
#encoding=utf8
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2016-08-18 18:24:05.771671
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import six

import collections
from collections import namedtuple
import numpy as np

import glob 
import math 

import re
import time

import gezi

def is_cn(word):
   return '\u4e00' <= item[0] <= '\u9fa5' 

def break_sentence(sentence, max_sent_len, additional=5):
  """
  For example, for a sentence with 70 words, supposing the the `max_sent_len'
  is 30, break it into 3 sentences.

  :param sentence: list[str] the sentence
  :param max_sent_len:
  :return:
  """
  ret = []
  cur = 0
  length = len(sentence)
  while cur < length:
    if cur + max_sent_len + additional >= length:
      ret.append(sentence[cur: length])
      break
    ret.append(sentence[cur: min(length, cur + max_sent_len)])
    cur += max_sent_len
  return ret

def add_start_end(w, start='<S>', end='</S>'):
  return [start] + list(w) + [end]

def str2scores(l):
  if ',' in l:
    # this is list save (list of list)
    return np.array([float(x.strip()) for x in l[1:-1].split(',')])
  else:
    # this numpy save (list of numpy array)
    return np.array([float(x.strip()) for x in l[1:-1].split(' ') if x.strip()])

def get_unmodify_minutes(file_):
  file_mod_time = os.stat(file_).st_mtime
  last_time = (time.time() - file_mod_time) / 60
  return last_time

# ## Well not work well, so fall back to use py2 bseg 
# def to_simplify(sentence):
#   sentence = gezi.langconv.Converter('zh-hans').convert(sentence).replace('馀', '余')
#   return sentence

# def parse_list_str(input, sep=','):
#   return np.array([float(x.strip()) for x in input[1:-1].split(sep) if x.strip()])

# https://stackoverflow.com/questions/43146528/how-to-extract-all-the-emojis-from-text
# def extract_emojis(sentence):
#   import emoji
#   allchars = [str for str in sentence]
#   l = [c for c in allchars if c in emoji.UNICODE_EMOJI]
#   return l

def extract_emojis(content):
  import emoji
  emojis_list=map(lambda x:''.join(x.split()),emoji.UNICODE_EMOJI.keys())
  r = re.compile('|'.join(re.escape(p) for p in emojis_list))
  return r.sub(r'',content)

def remove_emojis(sentence):
  import emoji
  allchars = [str for str in sentence]
  l = [c for c in allchars if c not in emoji.UNICODE_EMOJI]
  return ''.join(l)

def is_emoji(w):
  import emoji
  return w in emoji.UNICODE_EMOJI

def dict2namedtuple(thedict, name):
  thenametuple = namedtuple(name, [])
  for key, val in thedict.items():
    if not isinstance(key, str):
      msg = 'dict keys must be strings not {}'
      raise ValueError(msg.format(key.__class__))

    if not isinstance(val, dict):
      setattr(thenametuple, key, val)
    else:
      newname = dict2namedtuple(val, key)
      setattr(thenametuple, key, newname)

  return thenametuple

def csv(s):
  s = s.replace("\"", "\"\"") 
  s = "\"" + s + "\""
  return s

def get_weights(weights):
  if isinstance(weights, str):
    weights = map(float, weights.split(','))
  total_weights = sum(weights)
  #alpha = len(weights) / total_weights
  alpha = 1 / total_weights
  weights = [x * alpha for x in weights] 
  return weights

def probs_entropy(probs):
  e_x = [-p_x*math.log(p_x,2) for p_x in probs]
  entropy = sum(e_x)
  return entropy

def dist(x,y):   
  return np.sqrt(np.sum((x-y)**2))

def cosine(a, b):
  from numpy import dot
  from numpy.linalg import norm
  return dot(a, b)/(norm(a)*norm(b))

def softmax(x, axis=-1):
    mx = np.amax(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - mx)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    res = x_exp / x_sum
    return res

# https://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python
def sigmoid(x):
  s = 1/(1+np.exp(-x))
  return s

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  # what about 4 channels ?  TODO
  return np.array(image.getdata()).reshape(
      (im_height, im_width, -1)).astype(np.uint8)[:,:,:3]

def dirname(input):
  if os.path.isdir(input):
    return input
  else:
    dirname = os.path.dirname(input)
    if not dirname:
      dirname = '.'
    return dirname

def non_empty(file):
  return os.path.exists(file) and os.path.getsize(file) > 0

def merge_dicts(*dict_args):
  """
  Given any number of dicts, shallow copy and merge into a new dict,
  precedence goes to key value pairs in latter dicts.
  #https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression
  """
  result = {}
  for dictionary in dict_args:
    result.update(dictionary)
  return result

def norm(text):
  return text.strip().lower().replace('。', '')

def loggest_match(cns, vocab, encode_unk=False, unk_vocab_size=None, vocab_size=None):
  len_ = len(cns)
  for i in range(len_):
    w = ''.join(cns[:len_ - i])
    #for compat with c++ vocabulary
    if vocab.has(w):
      return vocab.id(w), cns[len_ - i:]
    elif unk_vocab_size:
      return gezi.hash(w) % unk_vocab_size + vocab_size, cns[len_ - i:] 
  if encode_unk:
    return vocab.unk_id(), cns[1:]
  else:
    return -1, cns[1:]

#TODO might use trie to speed up longest match segment
def loggest_match_seg(word, vocab, encode_unk=False):
  cns = gezi.get_single_cns(word)
  word_ids = []
  while True:
    id, cns = loggest_match(cns, vocab, encode_unk=encode_unk)
    if id != -1:
      word_ids.append(id)
    if not cns:
      break 
  return word_ids 

def index(l, val):
  try:
    return l.index(val)
  except Exception:
    return len(l)

def to_pascal_name(name):
  if not name or name[0].isupper():
    return name 
  return gnu2pascal(name)

def to_gnu_name(name): 
  if not name or name[0].islower():
    return name
  return pascal2gnu(name)

def pascal2gnu(name):
  """
  convert from AbcDef to abc_def
  name must be in pascal format
  """
  l = [name[0].lower()]
  for i in range(1, len(name)):
    if name[i].isupper():
      l.append('_')
      l.append(name[i].lower())
    else:
      l.append(name[i])
  return ''.join(l)

def gnu2pascal(name):
  """
  convert from abc_def to AbcDef
  name must be in gnu format
  """
  l = []
  l = [name[0].upper()]
  need_upper = False
  for i in range(1, len(name)):
    if name[i] == '_':
      need_upper = True
      continue
    else:  
      if need_upper:
        l.append(name[i].upper())
      else:
        l.append(name[i])
      need_upper = False
  return ''.join(l)

  return ''.join(l)

def is_gbk_luanma(text):
  return len(text) > len(text.decode('gb2312', 'ignore').encode('utf8'))

def gen_sum_list(l):
  l2 = [x for x in l]
  for i in range(1, len(l)):
    l2[i] += l2[i - 1]
  return l2

def add_one(d, word):
  if not word in d:
    d[word] = 1
  else:
    d[word] += 1

def pretty_floats(values):
  if not isinstance(values, (list, tuple)):
    values = [values]
  #return [float('{:.5f}'.format(x)) for x in values]
  return [float('{:e}'.format(x)) for x in values]
  #return ['{}'.format(x) for x in values]

def get_singles(l):
  """
  get signle elment as list, filter list
  """
  return [x for x in l if not isinstance(x, collections.Iterable)]

def is_single(item):
  return not isinstance(item, collections.Iterable)

def iterable(item):
  """
  be careful!  like string 'abc' is iterable! 
  you may need to use if not isinstance(values, (list, tuple)):
  """
  return isinstance(item, collections.Iterable)

def is_list_or_tuple(item):
  return isinstance(item, (list, tuple))

def get_value_name_list(values, names):
  return ['{}:{:.5f}'.format(x[0], x[1]) for x in zip(names, values)]


def batches(l, batch_size):
  """
  :param l:           list
  :param group_size:  size of each group
  :return:            Yields successive group-sized lists from l.
  """
  for i in range(0, len(l), batch_size):
    yield l[i: i + batch_size]

#@TODO better pad 
def pad(l, maxlen, mark=0):
  if isinstance(l, list):
    l.extend([mark] * (maxlen - len(l)))
    return l[:maxlen]
  elif isinstance(l, np.ndarray):
    return l
  else:
    raise ValueError('not support')

def nppad(l, maxlen):
  if maxlen > len(l):
    return np.lib.pad(l, (0, maxlen - len(l)), 'constant')
  else:
    return l[:maxlen]

def try_mkdir(dir):
  if not os.path.exists(dir):
    print('make new dir: [%s]'%dir, file=sys.stderr)
    os.makedirs(dir)

def get_dir(path):
  if os.path.isdir(path):
    return path 
  return os.path.dirname(path)

#@TODO perf?
def dedupe_list(l):
  #l_set = list(set(l))
  #l_set.sort(key = l.index)
  l_set = []
  set_  = set()
  for item in l:
    if item not in set_:
      set_.add(item)
      l_set.append(item)
  return l_set

#@TODO
def parallel_run(target, args_list, num_threads):
  record = []
  for thread_index in xrange(num_threads):
    process = multiprocessing.Process(target=target,args=args_list[thread_index])
    process.start()
    record.append(process)
  
  for process in record:
    process.join()

import threading
def multithreads_run(target, args_list):
  num_threads = len(args_list)
  threads = []
  for args in args_list:
    t = threading.Thread(target=target, args=args) 
    threads.append(t) 
  for t in threads:
    t.join()

#@TODO move to bigdata_util.py


#----------------file related
def is_glob_pattern(input):
  return '*' in input

def file_is_empty(path):
  return os.stat(path).st_size==0

def list_files(inputs):
  files = []
  inputs = inputs.split(',')
  for input in inputs:
    if not input or not input.strip():
      continue
    parts = []
    if os.path.isdir(input):
      parts = [os.path.join(input, f) for f in os.listdir(input)]
    elif os.path.isfile(input):
      parts = [input]
    else:
      parts = glob.glob(input)
    files += parts 

  files = [x for x in files if not (file_is_empty(x) or x.endswith('num_records.txt'))]
  return files

def sorted_ls(path, time_descending=True):
  mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
  return list(sorted(os.listdir(path), key=mtime, reverse=time_descending))

def list_models(model_dir, time_descending=True):
  """
  list all models in model_dir
  """
  files = [file for file in glob.glob('%s/model.ckpt-*'%(model_dir)) if not file.endswith('.meta')]
  files.sort(key=lambda x: os.path.getmtime(x), reverse=time_descending)
  return files 
  
#----------conf
def save_conf(con):
  file = '%s.py'%con.__name__
  out = open(file, 'w')
  for key,value in con.__dict__.items():
    if not key.startswith('__'):
      if not isinstance(value, str):
        result = '{} = {}\n'.format(key, value)
      else:
        result = '{} = \'{}\'\n'.format(key, value)
      out.write(result)

def write_to_txt(data, file):
  out = open(file, 'w')
  out.write('{}'.format(data))

def read_int_from(file, default_value=None):
  return int(open(file).readline().strip().split()[0]) if os.path.isfile(file) else default_value

def read_float_from(file, default_value=None):
  return float(open(file).readline().strip().split()[0]) if os.path.isfile(file) else default_value

def read_str_from(file, default_value=None):
  return open(file).readline().strip() if os.path.isfile(file) else default_value

def img_html(img):
  return '<p><a href={0} target=_blank><img src={0} height=200></a></p>\n'.format(img)

def text_html(text):
  return '<p>{}</p>'.format(text)

def thtml(text):
  return text_html(text)

#@TODO support *content 
def hprint(content):
  print('<p>', content,'</p>')

def imgprint(img):
  print(img_html(img))


def unison_shuffle(a, b):
  """
  http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
  """
  assert len(a) == len(b)
  try:
    from sklearn.utils import shuffle
    a, b = shuffle(a, b, random_state=0)
    return a, b
  except Exception:
    print('sklearn not installed! use numpy but is not inplace shuffle', file=sys.stderr)
    import numpy
    index = numpy.random.permutation(len(a))
    return a[index], b[index]

def finalize_feature(fe, mode='w', outfile='./feature_name.txt', sep='\n'):
  #print(fe.Str('\n'), file=sys.stderr)
  #print('\n'.join(['{}:{}'.format(i, fname) for i, fname in enumerate(fe.names())]), file=sys.stderr)
  #print(fe.Length(), file=sys.stderr)
  if mode == 'w':
    fe.write_names(file=outfile, sep=sep)
  elif mode == 'a':
    fe.append_names(file=outfile, sep=sep)

def write_feature_names(names, mode='a', outfile='./feature_name.txt', sep='\n'):
  out = open(outfile, mode)
  out.write(sep.join(names))
  out.write('\n')

def get_feature_names(file_):
  feature_names = []
  for line in open(file_):
    name = line.rstrip().split('#')[0].strip()
    if not name:
      continue
    feature_names.append(name)
  return feature_names

def read_feature_names(file_):
  feature_names = []
  for line in open(file_):
    name = line.rstrip().split('#')[0].strip()
    if not name:
      continue
    feature_names.append(name)
  return feature_names

def get_feature_names_dict(file_):
  feature_names_dict = {}
  index = 0
  for line in open(file_):
    name = line.rstrip().split('#')[0].strip()
    if not name:
      continue
    feature_names_dict[name] = index
    index += 1
  return feature_names_dict

def read_feature_names_dict(file_):
  feature_names_dict = {}
  index = 0
  for line in open(file_):
    name = line.rstrip().split('#')[0].strip()
    if not name:
      continue
    feature_names_dict[name] = index
    index += 1
  return feature_names_dict

def update_sparse_feature(feature, num_pre_features):
  features = feature.split(',')
  index_values = [x.split(':') for x in features]
  return ','.join(['{}:{}'.format(int(index) + num_pre_features, value) for index,value in index_values])

def merge_sparse_feature(fe1, fe2, num_fe1):
  if not fe1:
    return update_sparse_feature(fe2, num_fe1)
  if not fe2:
    return fe1 
  return ','.join([fe1, update_sparse_feature(fe2, num_fe1)])


#TODO move to other place
#http://blog.csdn.net/luo123n/article/details/9999481
def edit_distance(first,second):  
  if len(first) > len(second):  
    first,second = second,first  
  if len(first) == 0:  
    return len(second)  
  if len(second) == 0:  
    return len(first)  
  first_length = len(first) + 1  
  second_length = len(second) + 1  
  distance_matrix = [range(second_length) for x in range(first_length)]   
  for i in range(1,first_length):  
    for j in range(1,second_length):  
      deletion = distance_matrix[i-1][j] + 1  
      insertion = distance_matrix[i][j-1] + 1  
      substitution = distance_matrix[i-1][j-1]  
      if first[i-1] != second[j-1]:  
        substitution += 1  
      distance_matrix[i][j] = min(insertion,deletion,substitution)  
  return distance_matrix[first_length-1][second_length-1]  


import json
import gezi
def save_json(obj, filename):
  timer = gezi.Timer("Saving to {}...".format(filename))
  with open(filename, "w") as fh:
    json.dump(obj, fh)
  timer.print_elapsed()

def load_json(filename):
  timer = gezi.Timer("Loading {}...".format(filename))
  with open(filename) as fh:
    obj = json.load(fh)
  timer.print_elapsed()
  return obj

def read_json(filename):
  return load_json(filename)

def strip_suffix(s, suf):
  if s.endswith(suf):
    return s[:len(s)-len(suf)]
  return s

def log(text, array):
  """Prints a text message. And, optionally, if a Numpy array is provided it
  prints it's shape, min, and max values.
  """
  text = text.ljust(12)
  text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f} mean: {:10.5f} unique: {:5d} {}".format(
    str(array.shape),
    np.min(array) if array.size else "",
    np.max(array) if array.size else "",
    np.mean(array) if array.size else "",
    len(np.unique(array)) if array.size else "",
    array.dtype))
  print(text, file=sys.stderr)

def log_full(text, array):
  """Prints a text message. And, optionally, if a Numpy array is provided it
  prints it's shape, min, and max values.
  """
  log(text, array)
  print(np.unique(array), file=sys.stderr)

def env_has(name, val=1):
  return name in os.environ and int(os.environ[name]) == val

def env_get(name):
  if name in os.environ:
    return os.environ[name]
  else:
    return None

def env_set(name, val=1):
  os.environ[name] = str(val)

def env_val(name, default=None):
  return None if name not in os.environ else os.environ[name] 

def use_matplotlib(backend='Agg'):
  import matplotlib 
  matplotlib.use(backend) 

def decode(bytes_list):
  if not six.PY2:
    try:
      return np.array([x.decode() for x in bytes_list])
    except Exception:
      return bytes_list
  else:
    return bytes_list

def get_fold(total, num_folds, index):
  # index is None means all
  if index is None:
    return 0, total
  elif index < 0:
    return 0, 0
  assert num_folds
  fold_size = total // num_folds
  start = fold_size * index
  end = start + fold_size if index != num_folds - 1 else total
  return start, end

def is_fold(input, fold):
  input = strip_suffix(input, '.record')
  if fold is None:
    return False 
  try:
    # part-0003
    return int(re.split('-|_', input)[1]) == fold
  except Exception:
    # 3.record
    return int(re.split('.')[0]) == fold 

def to_list(item):
  if not isinstance(item, (list, tuple)):
    return [item]
  return item
