dir=./mount/temp/ai2018/sentiment/tfrecords/char.bert/
sh ./run-noemb-char-bert.sh $dir/vocab.txt
pushd .
cd ..
python ./read-records.py --type=dump --base=$dir
popd
