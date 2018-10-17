export BSEG=1 
export LD_LIBRARY_PATH=./lib:./:$LD_LIBRARY_PATH 
python gen-content.py | python gen-vocab.py --seg_method='phrase' --vocab_name='vocab.bseg.phrase.v2' --min_count=1
