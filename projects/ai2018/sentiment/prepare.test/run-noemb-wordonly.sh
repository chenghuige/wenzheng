vocab=$1
#python ./gen-records.py  --seg_method=basic --feed_single=0 --use_char=1 --vocab_=$vocab
python ./gen-records.py --input ./mount/data/ai2018/sentiment/sent/test.seg.txt --seg_method=basic --feed_single=0 --use_char=1 --vocab_=$vocab --word_only=1
#python ./gen-records.py --input ./mount/data/ai2018/sentiment/train.csv --start_index=1 --seg_method=basic --feed_single=0 --use_char=1 --vocab_=$vocab

#python ./gen-records.py --input ./mount/data/ai2018/sentiment/trans.en.csv --start_index=1 --seg_method=basic  --feed_single=0 --use_char=1
