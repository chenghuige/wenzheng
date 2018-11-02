python ./gen-lm-records.py --input ./mount/data/my-embedding/GloVe-sentiment-jieba/corpus/valid/ --use_char=1 --max_sentence_len=40 --tfrecord_dir tfrecord.len40
python ./gen-lm-records.py --input ./mount/data/my-embedding/GloVe-sentiment-jieba/corpus/train/ --use_char=1 --max_sentence_len=40 --tfrecord_dir tfrecord.len40
