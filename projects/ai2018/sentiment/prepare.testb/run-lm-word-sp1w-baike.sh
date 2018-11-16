vocab=./mount/temp/ai2018/sentiment/tfrecords/word.sp1w.ft/vocab.txt
python ./gen-lm-records.py --input ./mount/data/baidu/corpus.sp1w/train/ --use_char=1 --vocab_ $vocab --max_sentence_len=256 --source=baike
