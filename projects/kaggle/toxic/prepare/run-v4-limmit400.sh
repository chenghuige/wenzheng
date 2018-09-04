python3 ./gen-vocab.py --out_dir ./mount/temp/toxic/v4/tfrecords/glove-400/ --vocab_name vocab.full --min_count -1
python3 ./merge-glove.py --dir ./mount/temp/toxic/v4/tfrecords/glove-400/
python3 ./merge-charemb.py --dir ./mount/temp/toxic/v4/tfrecords/glove-400/
python3 ./gen-records.py --vocab ./mount/temp/toxic/v4/tfrecords/glove-400/vocab.txt --comment_limit 400
python3 ./gen-records.py --vocab ./mount/temp/toxic/v4/tfrecords/glove-400/vocab.txt --comment_limit 400 --input ~/data/kaggle/toxic/test.csv

