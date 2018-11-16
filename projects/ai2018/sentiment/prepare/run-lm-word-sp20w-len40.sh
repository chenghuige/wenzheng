vocab=./mount/temp/ai2018/sentiment/tfrecords/word.sp20w.ft/vocab.txt
python ./gen-lm-records.py --input ./mount/data/my-embedding/GloVe-sentiment-sentencepiece-20w/corpus/valid/ --use_char=1 --vocab_ $vocab --max_sentence_len=40 --tfrecord_dir tfrecord.len40
python ./gen-lm-records.py --input ./mount/data/my-embedding/GloVe-sentiment-sentencepiece-20w/corpus/train/ --use_char=1 --vocab_ $vocab --max_sentence_len=40 --tfrecord_dir tfrecord.len40
