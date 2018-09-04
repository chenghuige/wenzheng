export FAST=1
dir=./mount/temp/toxic/v15/tfrecords/glove.clean.lower/
python3 ./gen-vocab-parse.py --out_dir $dir --vocab_name vocab.full --min_count -1 --full_tokenizer=0 --max_ngrams=20000 --lower=1 --ngram_lower=1 --input ./mount/data/kaggle/toxic/train_cleaned.csv --test_input ./mount/data/kaggle/toxic/test_cleaned.csv 
python3 ./merge-emb.py --dir $dir --out_name glove.npy --min_count=10
#python ./merge-ngram-emb.py --dir $dir
python3 ./merge-charemb.py --dir $dir --min_count=10
cat $dir/vocab.txt | vocab2project.py > $dir/vocab.project 
python3 ./gen-records-parse.py --vocab $dir/vocab.txt  --input $dir/train.csv --lower=1 --ngram_lower=1
python3 ./gen-records-parse.py --vocab $dir/vocab.txt --input $dir/test.csv --lower=1 --ngram_lower=1

input=/home/gezi/data/kaggle/toxic/hate-speech-and-offensive-language-master/data/nontoxic.csv
python3 ./gen-vocab-parse.py --out_dir $dir --vocab_name vocab.full --min_count -1 --full_tokenizer=0  --name nontoxic.csv --input=$input --lower=1 --ngram_lower=1
python3 ./gen-records-parse.py --vocab $dir/vocab.txt --input $dir/nontoxic.csv --mode_ nontoxic --lower=1 --ngram_lower=1
export FAST=0

