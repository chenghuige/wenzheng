dir=./mount/temp/ai2018/reader/tfrecords/char.glove
emb=./mount/data/my-embedding.v1/GloVe-dureader-char/vectors.fix.txt
emb_dim=300

python ./merge-emb.py --input_vocab=$dir/vocab.ori.txt --emb=$emb --emb_dim=$emb_dim
