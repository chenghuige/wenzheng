base=./mount 

if [ $SRC ];
  then echo 'SRC:' $SRC 
else
  SRC='word.jieba.ft'
  echo 'use default SRC word.jieba.ft'
fi 

if [ $CELL ];
  then echo 'CELL:' $CELL 
else
  CELL='gru'
  echo 'use default CELL gru'
fi 
dir=$base/temp/ai2018/sentiment/tfrecords/lm/$SRC

fold=0
if [ $# == 1 ];
  then fold=$1
fi 
if [ $FOLD ];
  then fold=$FOLD
fi 

model_dir=$base/temp/ai2018/sentiment/model/lm/$SRC/tf.word.lm.rnet.$CELL.rand/
num_epochs=1

mkdir -p $model_dir/epoch 
cp $dir/vocab* $model_dir
cp $dir/vocab* $model_dir/epoch

exe=./lm-train.py 
if [ "$INFER" = "1"  ]; 
  then echo "INFER MODE" 
  exe=./infer.py 
  model_dir=$1
  fold=0
fi

if [ "$INFER" = "2"  ]; 
  then echo "VALID MODE" 
  exe=./infer.py 
  model_dir=$1
  fold=0
fi

python $exe \
        --use_char=1 \
        --concat_layers=1 \
        --recurrent_dropout=0 \
        --use_label_rnn=0 \
        --hop=1 \
        --rnn_no_padding=0 \
        --rnn_padding=1 \
        --model=RNet \
        --label_emb_height=20 \
        --use_label_att=1 \
        --use_self_match=1 \
        --vocab $dir/vocab.txt \
        --model_dir=$model_dir \
        --train_input=$dir/train/'*,' \
        --valid_input=$dir/valid/'*,' \
        --info_path=$dir/info.pkl \
        --emb_dim 300 \
        --finetune_word_embedding=1 \
        --batch_size 32 \
        --encoder_type=rnn \
        --cell=$CELL \
        --keep_prob=0.7 \
        --num_layers=1 \
        --rnn_hidden_size=400 \
        --encoder_output_method=topk,att \
        --eval_interval_steps 1000 \
        --metric_eval_interval_steps 1000 \
        --save_interval_steps 1000 \
        --save_interval_epochs=0.1 \
        --valid_interval_epochs=1 \
        --inference_interval_epochs=1 \
        --freeze_graph=1 \
        --optimizer=bert \
        --learning_rate=0.001 \
        --min_learning_rate=1e-5 \
        --warmup_steps=2000 \
        --num_epochs=$num_epochs \

