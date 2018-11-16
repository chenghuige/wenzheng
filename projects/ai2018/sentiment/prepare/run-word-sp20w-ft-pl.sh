dir=./mount/temp/ai2018/sentiment/tfrecords/word.sp20w.ft.pl
emb=./mount/data/my-embedding/fastText-sentiment-sentencepiece-20w/text.vec
emb_dim=300

#python ./merge-emb.py --input_vocab=$dir/vocab.ori.txt --emb=$emb --emb_dim=$emb_dim

sh ./run-noemb-pl.sh $dir/vocab.txt

#sh ./run-noemb-aug.sh $dir/vocab.txt seg

pushd .
cd ..
python ./read-records.py --type=dump --base=$dir  --use_soft_label=1
python read-records.py --base=$dir --type show_info --use_soft_label=1
popd
