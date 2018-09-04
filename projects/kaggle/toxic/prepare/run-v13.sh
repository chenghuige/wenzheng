dir=./mount/temp/toxic/v13/tfrecords/glove/
python3 ./gen-vocab-parse.py --out_dir $dir --vocab_name vocab.full --min_count -1 --full_tokenizer=0
python3 ./merge-emb.py --dir $dir --out_name glove.npy --min_count 10
python3 ./merge-charemb.py --dir $dir
cat $dir/vocab.txt | vocab2project.py > $dir/vocab.project 
python3 ./gen-records-parse.py --vocab $dir/vocab.txt  --input $dir/train.csv 
python3 ./gen-records-parse.py --vocab $dir/vocab.txt --input $dir/test.csv 

