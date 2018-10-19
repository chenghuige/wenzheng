export BSEG=1 
export LD_LIBRARY_PATH=./lib:./:$LD_LIBRARY_PATH 
python gen-content.py | python gen-vocab.py --seg_method='phrase_single_all' --vocab_name='vocab.bseg.phrase' --min_count=1
