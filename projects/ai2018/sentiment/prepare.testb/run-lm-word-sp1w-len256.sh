vocab=./mount/temp/ai2018/sentiment/tfrecords/word.sp1w.ft/vocab.txt
python ./gen-lm-records.py --input ./mount/data/my-embedding/GloVe-sentiment-sentencepiece-1w/corpus/valid/ --use_char=1 --vocab_ $vocab --max_sentence_len=256 --tfrecord_dir tfrecord.256
python ./gen-lm-records.py --input ./mount/data/my-embedding/GloVe-sentiment-sentencepiece-1w/corpus/train/ --use_char=1 --vocab_ $vocab --max_sentence_len=256 --tfrecord_dir tfrecord.256
