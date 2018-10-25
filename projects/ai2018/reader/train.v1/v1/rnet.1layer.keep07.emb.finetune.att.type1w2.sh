base=./mount
dir=$base/temp/ai2018/reader/tfrecord/

fold=0
if [ $# == 1 ];
  then fold=$1
fi 
if [ $FOLD ];
  then fold=$FOLD
fi 

model_dir=$base/temp/ai2018/reader/model/rnet.1layer.keep07.emb.finetune.att.type1w2/
num_epochs=10

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
        --type1_weight=2. \
        --model=Rnet \
        --att_combiner=gate \
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
        --batch_size 32 \
        --buckets 400 \
        --batch_sizes 32,16 \
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
        --decay_patience=1 \
        -decay_factor=0.8 \
        --num_epochs=$num_epochs \

