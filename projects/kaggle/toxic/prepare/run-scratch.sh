python3 ./gen-vocab.py --out_dir ./mount/temp/toxic/tfrecords/glove-scratch/ --vocab_name vocab.full --min_count -1
python3 ./merge-glove.py  --dir ./mount/temp/toxic/tfrecords/glove-scratch/ --type scratch
python3 ./gen-records.py  --vocab ./mount/temp/toxic/tfrecords/glove-scratch/vocab.txt 

