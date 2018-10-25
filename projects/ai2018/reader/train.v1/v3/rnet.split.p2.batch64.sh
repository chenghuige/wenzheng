base=./mount
dir=$base/temp/ai2018/reader/tfrecord/

fold=0
if [ $# == 1 ];
  then fold=$1
fi 
if [ $FOLD ];
  then fold=$FOLD
fi 

model_dir=$base/temp/ai2018/reader/model/v3/rnet.split.p2.batch64
num_epochs=20

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
        --split_type=1 \
        --model=Rnet \
        --att_combiner=sfu \
        --rcontent=1 \
        --use_type=1 \
        --vocab $dir/vocab.txt \
        --model_dir=$model_dir \
        --train_input=$dir/train/'*,' \
        --valid_input=$dir/valid/'*,' \
        --test_input=$dir/test/'*,' \
        --info_path=$dir/info.pkl \
        --word_embedding_file=$dir/emb.npy \
        --finetune_word_embedding=1 \
        --emb_dim 300 \
        --buckets 400 \
        --batch_sizes 64,32 \
        --batch_size 64 \
        --length_key passage \
        --encoder_type=rnn \
        --keep_prob=0.7 \
        --num_layers=1 \
        --rnn_hidden_size=100 \
        --encoder_output_method=att \
        --eval_interval_steps 1000 \
        --metric_eval_interval_steps 1000 \
        --save_interval_steps 1000 \
        --save_interval_epochs=1 \
        --valid_interval_epochs=1 \
        --inference_interval_epochs=1 \
        --freeze_graph=1 \
        --optimizer=adam \
        --learning_rate=0.001 \
        --decay_target=acc \
        --decay_patience=2 \
        --decay_factor=0.5 \
        --decay_start_epoch=2 \
        --num_epochs=$num_epochs \

