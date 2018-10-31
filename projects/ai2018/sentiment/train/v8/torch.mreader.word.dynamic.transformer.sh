base=./mount 

if [ $SRC ];
  then echo 'SRC:' $SRC 
else
  SRC='word.glove'
  echo 'use default SRC word.glove'
fi 
dir=$base/temp/ai2018/sentiment/tfrecords/$SRC

fold=0
if [ $# == 1 ];
  then fold=$1
fi 
if [ $FOLD ];
  then fold=$FOLD
fi 

model_dir=$base/temp/ai2018/sentiment/model/v8/$fold/$SRC/torch.mreader.word.dynamic.transformer/
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
        --dynamic_finetune=1 \
        --num_finetune_words=6000 \
        --num_finetune_chars=3000 \
        --use_char=1 \
        --concat_layers=0 \
        --recurrent_dropout=0 \
        --use_label_rnn=0 \
        --hop=1 \
        --att_combiner='sfu' \
        --rnn_no_padding=0 \
        --rnn_padding=1 \
        --model=MReader \
        --use_self_match=1 \
        --label_emb_height=20 \
        --fold=$fold \
        --use_label_att=1 \
        --use_self_match=1 \
        --vocab $dir/vocab.txt \
        --model_dir=$model_dir \
        --train_input=$dir/train/'*,' \
        --test_input=$dir/test/'*,' \
        --info_path=$dir/info.pkl \
        --emb_dim 300 \
        --word_embedding_file=$dir/emb.npy \
        --finetune_word_embedding=1 \
        --batch_size 32 \
        --buckets=500,1000 \
        --batch_sizes 32,16,8 \
        --length_key content \
        --encoder_type=transformer \
        --cell=gru \
        --keep_prob=0.7 \
        --num_layers=1 \
        --rnn_hidden_size=200 \
        --encoder_output_method=topk,att \
        --eval_interval_steps 1000 \
        --metric_eval_interval_steps 1000 \
        --save_interval_steps 1000 \
        --save_interval_epochs=1 \
        --valid_interval_epochs=1 \
        --inference_interval_epochs=1 \
        --freeze_graph=1 \
        --optimizer=adamax \
        --learning_rate=0.002 \
        --decay_target=loss \
        --decay_patience=1 \
        --decay_factor=0.8 \
        --decay_start_epoch_=2. \
        --num_epochs=$num_epochs \

