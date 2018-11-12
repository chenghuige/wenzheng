cd ./prepare/ 
sh run-noemb.sh 
cd ..
python ./read-records.py --type=dump
