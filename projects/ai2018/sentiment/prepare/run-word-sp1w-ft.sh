dir=./mount/temp/ai2018/sentiment/tfrecords/word.sp1w.ft
emb=./mount/data/my-embedding/fastText-sentiment-sentencepiece-1w/text.vec
emb_dim=300

python ./merge-emb.py --input_vocab=$dir/vocab.ori.txt --emb=$emb --emb_dim=$emb_dim

sh ./run-noemb.sh $dir/vocab.txt

#sh ./run-noemb-aug.sh $dir/vocab.txt seg

pushd .
cd ..
python ./read-records.py --type=dump --base=$dir
popd
