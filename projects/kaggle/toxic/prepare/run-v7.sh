dir=./mount/temp/toxic/v7/tfrecords/glove/
#python3 ./gen-vocab.py --out_dir ./mount/temp/toxic/v7/tfrecords/glove/ --vocab_name vocab.full --min_count -1
#python3 ./merge-glove.py --dir ./mount/temp/toxic/v7/tfrecords/glove/
#python3 ./merge-charemb.py --dir ./mount/temp/toxic/v7/tfrecords/glove/ 
cat $dir/vocab.txt | vocab2project.py > $dir/vocab.project 
#python3 ./gen-records.py --vocab ./mount/temp/toxic/v7/tfrecords/glove/vocab.txt 
#python3 ./gen-records.py --vocab ./mount/temp/toxic/v7/tfrecords/glove/vocab.txt --input ~/data/kaggle/toxic/test.csv

