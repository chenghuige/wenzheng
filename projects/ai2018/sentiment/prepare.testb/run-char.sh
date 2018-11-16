dir=./mount/temp/ai2018/sentiment/tfrecords/char
sh ./run-noemb-char.sh $dir/vocab.txt

pushd .
cd ..
python ./read-records.py --type=dump --base=$dir
python read-records.py --base=$dir --type show_info
popd
