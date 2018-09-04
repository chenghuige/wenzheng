python3 ./gen-vocab.py --out_dir ./mount/temp/toxic/tfrecords/glove/ --vocab_name vocab.full --min_count -1
python3 ./merge-glove.py 
python3 ./merge-charemb.py
python3 ./gen-records.py 
python3 ./gen-records.py --input ~/data/kaggle/toxic/test.csv

