python ./read-record.py \
    --train_input=../input/tfrecord/train \
    --valid_input=../input/tfrecord/valid \
    --batch_size=500 \
    --max_feat_len=100 \
    --feat_file_path=../input/feature_index \
    --field_file_path=../input/feat_fields.old
