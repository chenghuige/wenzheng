dir=./mount/temp/ai2018/sentiment/tfrecords/mix.bseg.glove.10epoch
emb=./mount/data/my-embedding/GloVe-sentiment-bseg-mix-10epoch/vectors.fix.txt
emb_dim=300

python ./merge-emb.py --input_vocab=$dir/vocab.ori.txt --emb=$emb --emb_dim=$emb_dim

sh ./run-noemb-mix.sh $dir/vocab.txt

pushd .
cd ..
python ./read-records.py --type=dump --base=$dir
popd
