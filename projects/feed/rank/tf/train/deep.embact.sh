model=Deep
python ./train.py \
    --emb_activation=relu \
    --model=$model \
    --num_epochs=3 \
    --eager=0 \
    --valid_interval_epochs=0.1 \
    --train_input=../input/train \
    --valid_input=../input/valid \
    --model_dir=../input/model/$model.embact \
    --batch_size=512 \
    --max_feat_len=100 \
    --feat_file_path=../input/feature_index \
    --field_file_path=../input/feat_fields.old
