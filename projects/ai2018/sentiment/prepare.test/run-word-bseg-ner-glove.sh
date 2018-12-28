dir=./mount/temp/ai2018/sentiment/tfrecords/word.bseg.ner.glove/
emb=./mount/data/my-embedding/GloVe-sentiment-bseg/vectors.fix.txt
emb_dim=300

#python ./merge-emb.py --input_vocab=$dir/vocab.ori.txt --emb=$emb --emb_dim=$emb_dim

sh ./run-noemb.sh $dir/vocab.txt

pushd .
cd ..
python ./read-records.py --type=dump --base=$dir 
python read-records.py --base=$dir --type show_info
popd
