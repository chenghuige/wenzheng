cd ./prepare/ 
sh run-unkvocab.sh 
cd ..
python ./read-records.py --type=dump
