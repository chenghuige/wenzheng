dir=./mount/temp/ai2018/reader/tfrecords/word.jieba.glove
emb=./mount/data/my-embedding.v1/GloVe-dureader-basic/vectors.fix.txt
emb_dim=300

python ./merge-emb.py --input_vocab=$dir/vocab.ori.txt --emb=$emb --emb_dim=$emb_dim --max_words=400000

sh ./run-noemb.sh $dir

#sh ./run-noemb-aug.sh $dir/vocab.txt seg

pushd .
cd ..
python ./read-records.py --type=dump --base=$dir
popd
