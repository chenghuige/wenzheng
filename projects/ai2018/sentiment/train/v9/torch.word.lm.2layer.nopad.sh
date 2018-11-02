base=./mount 

if [ $SRC ];
  then echo 'SRC:' $SRC 
else
  SRC='word.jieba'
  echo 'use default SRC word.jieba'
fi 
dir=$base/temp/ai2018/sentiment/tfrecords/lm/$SRC

fold=0
if [ $# == 1 ];
  then fold=$1
fi 
if [ $FOLD ];
  then fold=$FOLD
fi 

model_dir=$base/temp/ai2018/sentiment/model/lm/$SRC/torch.word.lm.2layer.nopad/
num_epochs=5

mkdir -p $model_dir/epoch 
cp $dir/vocab* $model_dir
cp $dir/vocab* $model_dir/epoch

exe=./torch-lm-train.py 
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
        --concat_layers=0 \
        --recurrent_dropout=0 \
        --use_label_rnn=0 \
        --hop=1 \
        --att_combiner='sfu' \
        --rnn_no_padding=1 \
        --rnn_padding=0 \
        --model=BiLanguageModel \
        --label_emb_height=20 \
        --use_label_att=0 \
        --use_self_match=0 \
        --vocab $dir/vocab.txt \
        --model_dir=$model_dir \
        --train_input=$dir/train/'*,' \
        --valid_input=$dir/valid/'*,' \
        --info_path=$dir/info.pkl \
        --emb_dim 300 \
        --word_embedding_file=$dir/emb.npy \
        --finetune_word_embedding=1 \
        --batch_size 32 \
        --encoder_type=rnn \
        --cell=gru \
        --keep_prob=0.7 \
        --num_layers=2 \
        --rnn_hidden_size=200 \
        --eval_interval_steps 1000 \
        --metric_eval_interval_steps 1000 \
        --save_interval_steps 1000 \
        --save_interval_epochs=0.1 \
        --valid_interval_epochs=1 \
        --inference_interval_epochs=1 \
        --freeze_graph=1 \
        --optimizer=adamax \
        --learning_rate=0.002 \
        --learning_rate_decay_factor=0.8 \
        --num_epochs_per_decay=0.1 \
        --num_epochs=$num_epochs \

