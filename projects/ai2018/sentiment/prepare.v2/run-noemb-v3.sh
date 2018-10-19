python ./gen-records.py  --simplify=1 --seg_method=basic --feed_single=0 --use_char=1
python ./gen-records.py --input ./mount/data/ai2018/sentiment/train.csv --simplify=1 --start_index=1 --seg_method=basic --feed_single=0 --use_char=1
python ./gen-records.py --input ./mount/data/ai2018/sentiment/test.csv --simplify=1 --seg_method=basic --feed_single=0 --use_char=1

#python ./gen-records.py --input ./mount/data/ai2018/sentiment/train.csv --simplify=0 --augument=1 --start_index=1 --seg_method=basic_digit
#python ./gen-records.py --simplify=0 --augument=1 --seg_method=basic_digit
#python ./gen-records.py --input ./mount/data/ai2018/sentiment/test.csv --simplify=0 --augument=1 --seg_method=basic_digit 

python ./gen-records.py --input ./mount/data/ai2018/sentiment/trans.en.csv --simplify=1 --start_index=1 --seg_method=basic  --feed_single=0 --use_char=1

#python ./gen-records.py --input ./mount/data/ai2018/sentiment/sentiment_classify_data/comment_raw_v2/canyin.csv --simplify=0 --start_index=0 --seg_method=basic_digit 
#python ./gen-records.py --input ./mount/data/ai2018/sentiment/dianping/dianping.csv --simplify=0 --start_index=0 --seg_method=basic_digit 

