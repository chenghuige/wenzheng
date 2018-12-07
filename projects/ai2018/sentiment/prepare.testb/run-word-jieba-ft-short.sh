dir=./mount/temp/ai2018/sentiment/tfrecords/word.jieba.ft.short
emb=./mount/data/my-embedding/fastText-sentiment-jieba/text.vec
emb_dim=300

#python ./merge-emb.py --input_vocab=$dir/vocab.ori.txt --emb=$emb --emb_dim=$emb_dim

vocab=$dir/vocab.txt
python ./gen-records.py --input ./mount/data/ai2018/sentiment/test.short.csv --seg_method=basic --feed_single=0 --use_char=1 --vocab_=$vocab --word_only=1

#sh ./run-noemb-aug.sh $dir/vocab.txt seg

pushd .
cd ..
python ./read-records.py --type=dump --base=$dir 
python read-records.py --base=$dir --type show_info
popd
