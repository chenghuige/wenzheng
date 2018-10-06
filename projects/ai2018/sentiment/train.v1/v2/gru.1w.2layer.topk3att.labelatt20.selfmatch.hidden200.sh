base=./mount
dir=$base/temp/ai2018/sentiment/tfrecord.1w.glove.dianping/

fold=0
if [ $# == 1 ];
  then fold=$1
fi 
if [ $FOLD ];
  then fold=$FOLD
fi 

model_dir=$base/temp/ai2018/sentiment/model/gru.1w.2layer.topk3att.labelatt20.selfmatch.hidden200
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

mode=train
if [ "$INFER" = "2"  ]; 
  then echo "VALID MODE" 
  #exe=./infer.py 
  mode=valid 
  model_dir=$1
  fold=0
fi

if [ "$INFER" = "3"  ]; 
  then echo "TEST MODE" 
  #exe=./infer.py 
  mode=test
  model_dir=$1
  fold=0
fi

if [ "$INFER" = "4"  ]; 
  then echo "VALID+TEST MODE" 
  #exe=./infer.py 
  mode=valid,test
  model_dir=$1
  fold=0
fi


python $exe \
        --use_self_match=1 \
        --label_emb_height=20 \
        --use_label_att=1 \
        --vocab $dir/vocab.txt \
        --model_dir=$model_dir \
        --train_input=$dir/train/'*,' \
        --valid_input=$dir/valid/'*,' \
        --test_input=$dir/test/'*,' \
        --info_path=$dir/info.pkl \
        --emb_dim 300 \
        --batch_size 32 \
        --encoder_type=rnn \
        --keep_prob=0.7 \
        --num_layers=2 \
        --rnn_hidden_size=200 \
        --encoder_output_method=topk,att \
        --top_k=3 \
        --eval_interval_steps 1000 \
        --metric_eval_interval_steps 1000 \
        --save_interval_steps 1000 \
        --save_interval_epochs=1 \
        --valid_interval_epochs=1 \
        --inference_interval_epochs=1 \
        --freeze_graph=1 \
        --optimizer=adam \
        --learning_rate=0.001 \
        --decay_target=f1 \
        --decay_patience=1 \
        --decay_factor=0.8 \
        --num_epochs=$num_epochs \
        --mode=$mode \

