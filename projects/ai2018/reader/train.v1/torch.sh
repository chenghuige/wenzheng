base=./mount
dir=$base/temp/ai2018/reader/tfrecord/

fold=0
if [ $# == 1 ];
  then fold=$1
fi 
if [ $FOLD ];
  then fold=$FOLD
fi 

model_dir=$base/temp/ai2018/reader/model/torch
num_epochs=20

mkdir -p $model_dir/epoch 
cp $dir/vocab* $model_dir
cp $dir/vocab* $model_dir/epoch

exe=./torch-train.py 
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


        #--model=MwAN \
        #--buckets 400 \
        #--length_key passage \
        #--batch_sizes 32,16 \
        #--buckets 200,350,800,1600,3200 \
        #--length_key rcontent \
        #--batch_sizes 32,16,8,4,2,1 \
        #--batch_size 32 \
python $exe \
        --model=Model \
        --use_type=1 \
        --rcontent=1 \
        --vocab=$dir/vocab.txt \
        --model_dir=$model_dir \
        --train_input=$dir/train/'*,' \
        --valid_input=$dir/valid/'*,' \
        --test_input=$dir/test/'*,' \
        --info_path=$dir/info.pkl \
        --emb_dim 300 \
        --length_key rcontent \
        --buckets 400 \
        --length_key rcontent \
        --batch_sizes 32,16 \
        --batch_size 32 \
        --encoder_type=rnn \
        --keep_prob=1. \
        --num_layers=1 \
        --rnn_hidden_size=100 \
        --encoder_output_method=max \
        --interval_steps 100 \
        --eval_interval_steps 1000 \
        --metric_eval_interval_steps 1000 \
        --save_interval_steps 1000 \
        --save_interval_epochs=1 \
        --valid_interval_epochs=1 \
        --inference_interval_epochs=1 \
        --freeze_graph=1 \
        --optimizer=adam \
        --learning_rate=0.002 \
        --decay_target=acc \
        --decay_patience=1 \
        --decay_factor=0.8 \
        --num_epochs=$num_epochs \

