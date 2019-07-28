model=WideDeep2
python ./train.py \
    --model=$model \
    --num_epochs=2 \
    --eager=0 \
    --valid_interval_epochs=0.5 \
    --train_input=../input/train \
    --valid_input=../input/valid \
    --model_dir=../input/model/$model.dw.bigbatch \
    --batch_size=40000 \
    --max_feat_len=100 \
    --optimizer=bert \
    --min_learning_rate=1e-6 \
    --warmup_steps=1000 \
    --learning_rate=0.002 \
    --feat_file_path=../input/feature_index \
    --field_file_path=../input/feat_fields.old
