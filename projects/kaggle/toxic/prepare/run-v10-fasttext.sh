dir=./mount/temp/toxic/v10/tfrecords/fasttext/
python3 ./gen-vocab-parse.py --out_dir $dir --vocab_name vocab.full --min_count -1 --full_tokenizer=0 --tokenizer_vocab ./mount/data/fasttext/fasttext-vocab.txt
python3 ./merge-emb.py --dir $dir --out_name glove.npy --min_count 10 --emb ./mount/data/fasttext/crawl-300d-2M.vec
python3 ./merge-charemb.py --dir $dir
cat $dir/vocab.txt | vocab2project.py > $dir/vocab.project 
python3 ./gen-records-parse.py --vocab $dir/vocab.txt  --input $dir/train.csv --tokenizer_vocab ./mount/data/fasttext/fasttext-vocab.txt 
python3 ./gen-records-parse.py --vocab $dir/vocab.txt --input $dir/test.csv  --tokenizer_vocab ./mount/data/fasttext/fasttext-vocab.txt

