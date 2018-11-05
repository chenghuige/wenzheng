dir=$1 
vocab=$dir/vocab.txt 

python ./gen-records.py --input $dir/test.json  --use_char=1 --vocab_=$vocab
python ./gen-records.py --input $dir/train.corpus --use_char=1 --vocab_=$vocab

#python ./gen-records.py --input ./mount/data/ai2018/sentiment/trans.en.csv --start_index=1 --seg_method=basic  --feed_single=0 --use_char=1
