dir=./mount/temp/ai2018/reader/tfrecords/char.ft 
emb=./mount/data/my-embedding.v1/fastText-dureader-char/text.vec
emb_dim=300

python ./merge-emb.py --input_vocab=$dir/vocab.ori.txt --emb=$emb --emb_dim=$emb_dim
