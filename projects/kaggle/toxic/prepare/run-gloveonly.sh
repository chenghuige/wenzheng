python3 ./gen-vocab.py --out_dir ./mount/temp/toxic/tfrecords/glove-only/ --vocab_name vocab.full --min_count -1
python3 ./merge-glove.py  --dir ./mount/temp/toxic/tfrecords/glove-only/ --type only
python3 ./gen-records.py  --vocab ./mount/temp/toxic/tfrecords/glove-only/vocab.txt 
python3 ./gen-records.py  --vocab ./mount/temp/toxic/tfrecords/glove-only/vocab.txt --input ~/data/kaggle/toxic/test.csv
