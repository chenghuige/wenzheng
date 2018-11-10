base=./mount

if [ $SRC ];
  then echo 'SRC:' $SRC 
else
    SRC='word.sp1w.ft'
    echo 'use default SRC word.sp1w.ft'
fi 
dir=$base/temp/ai2018/sentiment/tfrecords/$SRC

fold=0
if [ $# == 1 ];
  then fold=$1
fi 
if [ $FOLD ];
  then fold=$FOLD
fi 

model_dir=$base/temp/ai2018/sentiment/model/v10/$fold/$SRC/tf.char.transformer.2/
num_epochs=30

mkdir -p $model_dir/epoch 
cp $dir/vocab* $model_dir
cp $dir/vocab* $model_dir/epoch

exe=./train.py 
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
        --num_finetune_words=3000 \
        --num_finetune_chars=3000 \
        --model=Transformer \
        --fold=$fold \
        --vocab $dir/vocab.txt \
        --model_dir=$model_dir \
        --train_input=$dir/train/'*,' \
        --test_input=$dir/test/'*,' \
        --info_path=$dir/info.pkl \
        --emb_dim 300 \
        --word_embedding_file=$dir/emb.npy \
        --finetune_word_embedding=1 \
        --batch_size 32 \
        --content_limit=512 \
        --buckets=128,256,320,512 \
        --batch_sizes 32,16,12,6,2 \
        --length_key content \
        --encoder_output_method=topk,att \
        --eval_interval_steps 1000 \
        --metric_eval_interval_steps 1000 \
        --save_interval_steps 1000 \
        --save_interval_epochs=1 \
        --valid_interval_epochs=1 \
        --inference_interval_epochs=1 \
        --freeze_graph=1 \
        --optimizer=bert \
        --learning_rate=1e-4 \
        --min_learning_rate=5e-6 \
        --num_decay_epochs=5 \
        --warmup_steps=2000 \
        --num_epochs=$num_epochs \

