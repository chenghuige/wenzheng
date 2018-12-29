base=./mount 

if [ $CELL ];
  then echo 'CELL:' $CELL 
else
  CELL='lstm'
  echo 'use default CELL lstm'
fi 
dir=$base/temp/ai2018/binary/tfrecord/

model_dir=$base/temp/ai2018/binary/model/base/
num_epochs=15

mkdir -p $model_dir/epoch 
cp $dir/vocab* $model_dir
cp $dir/vocab* $model_dir/epoch

exe=./torch-train.py 
if [ "$INFER" = "1"  ]; 
  then echo "INFER MODE" 
  exe=./$exe 
  model_dir=$1
  fold=0
fi

if [ "$INFER" = "2"  ]; 
  then echo "VALID MODE" 
  exe=./$exe 
  model_dir=$1
  fold=0
fi

python $exe \
        --lm_path=$base/temp/ai2018/sentiment/model/lm/word.jieba.ft.long/torch.word.lm.nopad.$CELL.hidden400/latest.pyt \
        --use_char=1 \
        --concat_layers=0 \
        --recurrent_dropout=0 \
        --use_label_rnn=0 \
        --hop=1 \
        --att_combiner='sfu' \
        --rnn_no_padding=1 \
        --rnn_padding=0 \
        --model=MReader \
        --use_label_att=0 \
        --use_self_match=1 \
        --vocab $dir/vocab.txt \
        --model_dir=$model_dir \
        --train_input=$dir/train/'*,' \
        --valid_input=$dir/valid/'*,' \
        --info_path=$dir/info.pkl \
        --emb_dim 300 \
        --word_embedding_file=$dir/emb.npy \
        --finetune_word_embedding=1 \
        --batch_size 32 \
        --buckets=500,1000 \
        --batch_sizes 32,16,8 \
        --length_key content \
        --encoder_type=rnn \
        --cell=$CELL \
        --keep_prob=0.7 \
        --num_layers=2 \
        --rnn_hidden_size=400 \
        --encoder_output_method=topk,att \
        --eval_interval_steps 1000 \
        --metric_eval_interval_steps=1000 \
        --save_interval_steps 1000 \
        --save_interval_epochs=100 \
        --valid_interval_epochs=1 \
        --inference_interval_epochs=1 \
        --optimizer=bert \
        --learning_rate=0.002 \
        --min_learning_rate=1e-5 \
        --num_decay_epochs=5 \
        --warmup_steps=2000 \
        --num_epochs=$num_epochs \

