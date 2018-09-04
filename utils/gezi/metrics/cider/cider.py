# Filename: cider.py
#
# Description: Describes the class to compute the CIDEr (Consensus-Based Image Description Evaluation) Metric 
#               by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>
from __future__ import print_function

from .cider_scorer import CiderScorer
import pdb
from collections import defaultdict

#need to install dill, which is used to dump and load defaultdict
try:
  import dill
except Exception:
  pass
  
import sys

class Cider:
    """
    Main Class to compute the CIDEr metric 

    """
    def __init__(self, test=None, refs=None, n=4, sigma=6.0, document_frequency=None, ref_len=None):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma

        #print('document_frequency', document_frequency, 'ref_len', ref_len, file=sys.stderr)
        if document_frequency is None:
            self._doucument_frequency = defaultdict(float)  
        elif isinstance(document_frequency, str):
            self._doucument_frequency = dill.load(open(document_frequency))
        else:
            self._doucument_frequency = document_frequency

        if isinstance(ref_len, str):
            self._ref_len = float(open(ref_len).readline().strip())
        else:
            self._ref_len = ref_len

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus 
        """

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma, 
                                   document_frequency=self._doucument_frequency, 
                                   ref_len=self._ref_len)

        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

            cider_scorer += (hypo[0], ref)

        (score, scores) = cider_scorer.compute_score()

        return score, scores

    def method(self):
        return "CIDEr"