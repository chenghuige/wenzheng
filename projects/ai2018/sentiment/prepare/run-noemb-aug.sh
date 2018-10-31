vocab=$1
name=$2
python ./gen-records.py  --seg_method=basic --feed_single=0 --use_char=1 --vocab_=$vocab --mode_ aug.$name.valid --mode aug.$name.train
python ./gen-records.py --input ./mount/data/ai2018/sentiment/test.csv --seg_method=basic --feed_single=0 --use_char=1 --vocab_=$vocab --mode_ aug.$name.test --mode aug.$name.test
python ./gen-records.py --input ./mount/data/ai2018/sentiment/train.csv --start_index=1 --seg_method=basic --feed_single=0 --use_char=1 --vocab_=$vocab --mode_ aug.$name.train --mode aug.$name.train

#python ./gen-records.py --input ./mount/data/ai2018/sentiment/trans.en.csv --start_index=1 --seg_method=basic  --feed_single=0 --use_char=1
