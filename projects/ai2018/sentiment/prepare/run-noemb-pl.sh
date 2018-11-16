vocab=$1
python ./gen-records.py  --seg_method=basic --feed_single=0 --use_char=1 --vocab_=$vocab --use_soft_label_=1 --word_only=1
python ./gen-records.py --input ./mount/data/ai2018/sentiment/train.csv --start_index=1 --seg_method=basic --feed_single=0 --use_char=1 --vocab_=$vocab --use_soft_label_=1 --word_only=1
python ./gen-records.py --input /home/gezi/temp/ai2018/sentiment/p40/model.csv/v11/submit.testa/ensemble.infer.debug.csv --seg_method=basic --feed_single=0 --use_char=1 --vocab_=$vocab --num_records_=10 --use_soft_label_=1 --is_soft_label=1 --word_only=1

#python ./gen-records.py --input ./mount/data/ai2018/sentiment/trans.en.csv --start_index=1 --seg_method=basic  --feed_single=0 --use_char=1
