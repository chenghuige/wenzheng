# coding: utf-8
import melt
# http://docs.fast.ai/text.html
from fastai.text import *
import html
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn import model_selection
import re
from functools import partial
from torch import optim

IMDB_PATH = Path('/home/gezi/data/imdb_sample/')


df = pd.read_csv(IMDB_PATH/'train.csv', header=None)
df.head()


classes = read_classes(IMDB_PATH/'classes.txt')
classes[0], classes[1]

data_lm = text_data_from_csv(Path(IMDB_PATH), data_func=lm_data)
data_clas = text_data_from_csv(Path(IMDB_PATH), data_func=classifier_data, vocab=data_lm.train_ds.vocab)

#download_wt103_model()

#learn = RNNLearner.language_model(data_lm, pretrained_fnames=['lstm_wt103', 'itos_wt103'], drop_mult=0.5)
#learn.fit_one_cycle(1, 1e-2)

# default bptt 70 will 
learn = RNNLearner.language_model(data_lm, bptt=30, pretrained_fnames=['lstm_wt103', 'itos_wt103'])
learn.unfreeze()
learn.fit(2, slice(1e-4,1e-2))

learn.save_encoder('enc') 

learn = RNNLearner.classifier(data_clas)
learn.load_encoder('enc')
learn.fit(3, 1e-3)

