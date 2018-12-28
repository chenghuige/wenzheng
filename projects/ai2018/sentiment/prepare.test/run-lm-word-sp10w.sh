vocab=./mount/temp/ai2018/sentiment/tfrecords/word.sp10w.ft/vocab.txt
python ./gen-lm-records.py --input ./mount/data/my-embedding/GloVe-sentiment-sentencepiece-10w/corpus/valid/ --use_char=1 --vocab_ $vocab
python ./gen-lm-records.py --input ./mount/data/my-embedding/GloVe-sentiment-sentencepiece-10w/corpus/train/ --use_char=1 --vocab_ $vocab
