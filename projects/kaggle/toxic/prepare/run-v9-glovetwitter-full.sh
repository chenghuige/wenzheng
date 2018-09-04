dir=./mount/temp/toxic/v9/tfrecords/glove.twitter.full/
python3 ./gen-vocab-parse.py --out_dir $dir --vocab_name vocab.full --min_count -1 --full_tokenizer=1 --is_twitter=1 --tokenizer_vocab ./mount/data/glove/glove.twitter.27B.200d.txt 
python3 ./merge-emb.py --dir $dir --out_name glove.npy --min_count 10 --emb ./mount/data/glove/glove.twitter.27B.200d.txt --emb_dim 200
python3 ./merge-charemb.py --dir $dir
cat $dir/vocab.txt | vocab2project.py > $dir/vocab.project 
python3 ./gen-records-parse.py --vocab $dir/vocab.txt  --input $dir/train.csv 
python3 ./gen-records-parse.py --vocab $dir/vocab.txt --input $dir/test.csv 

