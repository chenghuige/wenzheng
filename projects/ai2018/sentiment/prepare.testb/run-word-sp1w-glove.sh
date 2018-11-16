dir=./mount/temp/ai2018/sentiment/tfrecords/word.sp1w.glove
emb=./mount/data/my-embedding/GloVe-sentiment-sentencepiece-1w/vectors.fix.txt
emb_dim=300

#python ./merge-emb.py --input_vocab=$dir/vocab.ori.txt --emb=$emb --emb_dim=$emb_dim

sh ./run-noemb-wordonly.sh $dir/vocab.txt

#sh ./run-noemb-aug.sh $dir/vocab.txt seg

pushd .
cd ..
python ./read-records.py --type=dump --base=$dir
python read-records.py --base=$dir --type show_info
popd
