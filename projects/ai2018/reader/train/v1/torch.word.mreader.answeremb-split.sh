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
  CELL=gru
  echo 'use default CELL gru'
fi 


dir=$base/temp/ai2018/reader/tfrecords/$SRC

fold=0
if [ $# == 1 ];
  then fold=$1
fi 
if [ $FOLD ];
  then fold=$FOLD
fi 

model_dir=$base/temp/ai2018/reader/model/v1/$fold/$SRC/torch.word.mreader.anseremb.split.$CELL/
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

python $exe \
        --use_answer_emb=1 \
        --split_type=1 \
        --use_type_emb=0 \
        --use_type=0 \
        --rcontent=1 \
        --dynamic_finetune=1 \
        --num_finetune_words=6000 \
        --num_finetune_chars=3000 \
        --use_char=1 \
        --concat_layers=0 \
        --recurrent_dropout=0 \
        --use_label_rnn=0 \
        --hop=2 \
        --att_combiner='sfu' \
        --rnn_no_padding=0 \
        --rnn_padding=1 \
        --model=MReader \
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
        --buckets=500,1000,2000 \
        --batch_sizes 32,16,8,4 \
        --length_key rcontent \
        --encoder_type=rnn \
        --cell=$CELL \
        --keep_prob=0.7 \
        --num_layers=1 \
        --rnn_hidden_size=100 \
        --encoder_output_method=max \
        --eval_interval_steps 1000 \
        --metric_eval_interval_steps 1000 \
        --save_interval_steps 1000 \
        --save_interval_epochs=1 \
        --valid_interval_epochs=1 \
        --inference_interval_epochs=1 \
        --freeze_graph=1 \
        --optimizer=adamax \
        --learning_rate=0.002 \
        --decay_target=acc \
        --decay_patience=1 \
        --decay_factor=0.8 \
        --decay_start_epoch_=2. \
        --num_epochs=$num_epochs \

