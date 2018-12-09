base=./mount

if [ $SRC ];
  then echo 'SRC:' $SRC 
else
  SRC='char.nbert'
  echo 'use default SRC char.nbert'
fi 
dir=$base/temp/ai2018/sentiment/tfrecords/$SRC

fold=0
if [ $# == 1 ];
  then fold=$1
fi 
if [ $FOLD ];
  then fold=$FOLD
fi 

model_dir=$base/temp/ai2018/sentiment/model/v14/$fold/$SRC/tf.char.transformer.nbert.finetune.4gpu.5epoch.3layer/
num_epochs=5

mkdir -p $model_dir/epoch 
cp $dir/vocab* $model_dir
cp $dir/vocab* $model_dir/epoch

exe=./train.py 
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

# use 4 gpu to run if gtx1080ti FIXME now can only batc_size 2 * 4 .. why?
CUDA_VISIBLE_DEVICES=0,1,2,3 python $exe \
        --bert_num_layers=3 \
        --bert_lr_ratio=1. \
        --bert_dir=$base/data/my-embedding/bert-char/ckpt/500000 \
        --num_finetune_words=3000 \
        --model=Transformer \
        --fold=$fold \
        --vocab $dir/vocab.txt \
        --model_dir=$model_dir \
        --train_input=$dir/train/'*,' \
        --info_path=$dir/info.pkl \
        --emb_dim 300 \
        --finetune_word_embedding=1 \
        --batch_size=32 \
        --batch_size_per_gpu=0 \
        --length_key content \
        --encoder_output_method=last \
        --eval_interval_steps 1000 \
        --metric_eval_interval_steps 1000 \
        --save_interval_steps 1000 \
        --save_interval_epochs=1 \
        --valid_interval_epochs=1 \
        --inference_interval_epochs=1 \
        --freeze_graph=1 \
        --optimizer=bert \
        --learning_rate=8e-5 \
        --min_learning_rate=1e-7 \
        --num_epochs=$num_epochs \

