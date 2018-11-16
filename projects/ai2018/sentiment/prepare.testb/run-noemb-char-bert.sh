vocab=$1
#python ./gen-records.py --use_char=0 --vocab_=$vocab --fixed_vocab=1 --start_mark='[CLS]' --end_mark='[SEP]' --unk_word='[UNK]'
python ./gen-records.py --input ./mount/data/ai2018/sentiment/test.csv --use_char=0 --vocab_=$vocab --fixed_vocab=1 --start_mark='[CLS]' --end_mark='[SEP]' --unk_word='[UNK]'
#python ./gen-records.py --input ./mount/data/ai2018/sentiment/train.csv --start_index=1 --use_char=0 --vocab_=$vocab --fixed_vocab=1 --start_mark='[CLS]' --end_mark='[SEP]' --unk_word='[UNK]'

#python ./gen-records.py --input ./mount/data/ai2018/sentiment/trans.en.csv --start_index=1 --use_char=1
