#import tensorflow as tf

#------------new interface 
from melt.seq2seq.decoder import *
from melt.seq2seq.helper import *  
from melt.seq2seq.basic_decoder import * 
from melt.seq2seq.logprobs_decoder import * 
from melt.seq2seq.beam_search_decoder import *
from melt.seq2seq.attention_wrapper import * 

#------------old interface, depreciated
from melt.seq2seq.attention_decoder_fn import attention_decoder_fn_inference
from melt.seq2seq.attention_decoder_fn import attention_decoder_fn_train
from melt.seq2seq.attention_decoder_fn import prepare_attention

from melt.seq2seq.loss import *

from melt.seq2seq.decoder_fn import *
from melt.seq2seq.beam_decoder_fn import *
from melt.seq2seq.attention_decoder_fn import * 
from melt.seq2seq.seq2seq import *

from melt.seq2seq.beam_decoder import *

from melt.seq2seq.beam_search import *

