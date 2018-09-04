dir=./mount/temp/toxic/v9/tfrecords/glove/
python3 ./gen-vocab.py --out_dir $dir --vocab_name vocab.full --min_count -1
python3 ./merge-emb.py --dir $dir --out_name glove.npy
python3 ./merge-charemb.py --dir $dir
cat $dir/vocab.txt | vocab2project.py > $dir/vocab.project 
python3 ./gen-records.py --vocab $dir/vocab.txt  
python3 ./gen-records.py --vocab $dir/vocab.txt --input /home/gezi/data/kaggle/toxic/test.csv 

