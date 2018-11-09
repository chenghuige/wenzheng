BERT_BASE_DIR=/home/gezi/data/word-emb/chinese_L-12_H-768_A-12
python /home/gezi/mine/wenzheng/third/bert/create_pretraining_data.py \
  --input_file=/home/gezi/data/my-embedding/bert-char/text.bert.small \
  --output_file=/home/gezi/data/my-embedding/bert-char/tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
