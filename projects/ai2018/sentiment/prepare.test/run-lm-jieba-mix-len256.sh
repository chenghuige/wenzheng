vocab=./mount/temp/ai2018/sentiment/tfrecords/mix.jieba.ft/vocab.txt
python ./gen-lm-records.py --input ./mount/data/my-embedding/GloVe-sentiment-jieba-mix/corpus/valid/ --use_char=1 --max_sentence_len=256 --tfrecord_dir tfrecord.len256 --vocab_ $vocab
python ./gen-lm-records.py --input ./mount/data/my-embedding/GloVe-sentiment-jieba-mix/corpus/train/ --use_char=1 --max_sentence_len=256 --tfrecord_dir tfrecord.len256 --vocab_ $vocab
