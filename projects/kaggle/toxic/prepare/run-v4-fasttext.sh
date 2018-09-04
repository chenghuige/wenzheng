#mkdir -p ./mount/temp/toxic/v4/tfrecords/fasttext/
#cp ./mount/temp/toxic/v4/tfrecords/glove/*vocab*.txt ./mount/temp/toxic/v4/tfrecords/fasttext
python3 ./gen-vocab.py --out_dir ./mount/temp/toxic/v4/tfrecords/fasttext --vocab_name vocab.full --min_count -1 --tokenizer_vocab ./mount/data/fasttext/fasttext-vocab.txt
python3 ./merge-emb.py --dir ./mount/temp/toxic/v4/tfrecords/fasttext/ --emb ./mount/data/fasttext/crawl-300d-2M.vec 
python3 ./merge-charemb.py --dir ./mount/temp/toxic/v4/tfrecords/fasttext/ 
python3 ./gen-records.py --vocab ./mount/temp/toxic/v4/tfrecords/fasttext/vocab.txt --tokenizer_vocab ./mount/data/fasttext/fasttext-vocab.txt
python3 ./gen-records.py --vocab ./mount/temp/toxic/v4/tfrecords/fasttext/vocab.txt --input ~/data/kaggle/toxic/test.csv --tokenizer_vocab ./mount/data/fasttext/fasttext-vocab.txt

