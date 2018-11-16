dir=./mount/temp/ai2018/sentiment/tfrecords/char.nbert/
sh ./run-noemb-char-nbert.sh $dir/vocab.txt
pushd .
cd ..
python ./read-records.py --type=dump --base=$dir
popd
