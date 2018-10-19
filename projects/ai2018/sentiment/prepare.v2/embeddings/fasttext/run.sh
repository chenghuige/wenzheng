#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

RESULTDIR=./
DATADIR=./

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"

make

./fasttext skipgram -input "${DATADIR}"/text -output "${RESULTDIR}"/text -lr 0.025 -dim 300 \
  -ws 5 -epoch 10 -minCount 20 -neg 5 -loss ns -bucket 2000000 \
  -minn 2 -maxn 3 -thread 4 -t 1e-4 -lrUpdateRate 100

cut -f 1,2 "${DATADIR}"/rw/rw.txt | awk '{print tolower($0)}' | tr '\t' '\n' > "${DATADIR}"/queries.txt

cat "${DATADIR}"/queries.txt | ./fasttext print-word-vectors "${RESULTDIR}"/text.bin > "${RESULTDIR}"/vectors.txt

python eval.py -m "${RESULTDIR}"/vectors.txt -d "${DATADIR}"/rw/rw.txt
