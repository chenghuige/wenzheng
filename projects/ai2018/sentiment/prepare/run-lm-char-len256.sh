vocab=./mount/temp/ai2018/sentiment/tfrecords/char.ft/vocab.txt
python ./gen-lm-records.py --input ./mount/data/my-embedding/GloVe-sentiment-char/corpus/valid/ --use_char=0 --vocab_ $vocab --max_sentence_len=256 --tfrecord_dir tfrecord.len256
python ./gen-lm-records.py --input ./mount/data/my-embedding/GloVe-sentiment-char/corpus/train/ --use_char=0 --vocab_ $vocab --max_sentence_len=256 --tfrecord_dir tfrecord.len256
