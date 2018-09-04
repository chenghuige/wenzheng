python3 ./gen-vocab.py --out_dir ./mount/temp/toxic/v1/tfrecords/glove/ --vocab_name vocab.full --min_count -1
python3 merge-glove.py \
    --dir ./mount/temp/toxic/v1/tfrecords/glove/ \
    --include_non_match=0

python3 ./gen-records.py  --vocab ./mount/temp/toxic/v1/tfrecords/glove/vocab.txt 

