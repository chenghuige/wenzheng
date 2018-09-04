export FAST=1
dir=./mount/temp/toxic/v16/tfrecords/fasttext/
python3 ./gen-vocab-parse.py --out_dir $dir --vocab_name vocab.full --min_count -1 --full_tokenizer=0 --tokenizer_vocab ./mount/data/fasttext/fasttext-vocab.txt --max_ngrams=20000
python3 ./merge-emb.py --dir $dir --out_name glove.npy --min_count 10 --emb ./mount/data/fasttext/crawl-300d-2M.vec
python3 ./merge-charemb.py --dir $dir --min_count 10 
cat $dir/vocab.txt | vocab2project.py > $dir/vocab.project 
python3 ./gen-records-parse.py --vocab $dir/vocab.txt  --input $dir/train.csv 
python3 ./gen-records-parse.py --vocab $dir/vocab.txt --input $dir/test.csv 

input=/home/gezi/data/kaggle/toxic/hate-speech-and-offensive-language-master/data/nontoxic.csv
python3 ./gen-vocab-parse.py --out_dir $dir --vocab_name vocab.full --min_count -1 --full_tokenizer=0  --name nontoxic.csv --input=$input --tokenizer_vocab ./mount/data/fasttext/fasttext-vocab.txt
python3 ./gen-records-parse.py --vocab $dir/vocab.txt --input $dir/nontoxic.csv --mode_ nontoxic --weight=0.1

input=/home/gezi/data/kaggle/toxic/hate-speech-and-offensive-language-master/data/hate_label.csv
python3 ./gen-vocab-parse.py --out_dir $dir --vocab_name vocab.full --min_count -1 --full_tokenizer=0  --name hate.csv --input=$input --tokenizer_vocab ./mount/data/fasttext/fasttext-vocab.txt --lower=1 --ngram_lower=1
python3 ./gen-records-parse.py --vocab $dir/vocab.txt --input $dir/hate.csv --mode_ hate --lower=1 --ngram_lower=1 --weight=0.1

input=/home/gezi/data/kaggle/toxic/white/white_label.csv
python3 ./gen-vocab-parse.py --out_dir $dir --vocab_name vocab.full --min_count -1 --full_tokenizer=0  --name white.csv --input=$input --tokenizer_vocab ./mount/data/fasttext/fasttext-vocab.txt --lower=1 --ngram_lower=1
python3 ./gen-records-parse.py --vocab $dir/vocab.txt --input $dir/white.csv --mode_ white --lower=1 --ngram_lower=1 --weight=0.1

export FAST=0

