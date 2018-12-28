vocab=$1
#python ./gen-records.py  --seg_method=basic_digit --feed_single=1 --use_char=0 --vocab_=$vocab --mode_ aug.char.valid --mode aug.char.train
python ./gen-records.py --input ./mount/data/ai2018/sentiment/test.csv --seg_method=basic_digit --feed_single=1 --use_char=0 --vocab_=$vocab --mode_ aug.char.test --mode aug.char.test
#python ./gen-records.py --input ./mount/data/ai2018/sentiment/train.csv --start_index=1 --seg_method=basic_digit --feed_single=1 --use_char=0 --vocab_=$vocab --mode_ aug.char.train --mode aug.char.train

#python ./gen-records.py --input ./mount/data/ai2018/sentiment/trans.en.csv --start_index=1 --seg_method=basic  --feed_single=0 --use_char=1
