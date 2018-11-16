vocab=./mount/temp/ai2018/sentiment/tfrecords/word.jieba.tx/vocab.txt
python ./gen-lm-records.py --input ./mount/data/my-embedding/GloVe-sentiment-jieba/corpus.tx/train/ --use_char=1 --max_sentence_len=40 --tfrecord_dir tfrecord.len40 --vocab_ $vocab
python ./gen-lm-records.py --input ./mount/data/my-embedding/GloVe-sentiment-jieba/corpus.tx/valid/ --use_char=1 --max_sentence_len=40 --tfrecord_dir tfrecord.len40 --vocab_ $vocab
