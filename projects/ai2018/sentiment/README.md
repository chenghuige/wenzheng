# generate tfrecord  
go to prepare, sh run.sh gen valid/test/train   
# dump infos.pkl  
python ./read-records.py --type dump   

or just sh prepare.sh to do above, be sure you use correct tfrecord folder (using ln -s)  and vectors.txt if use word pretrain emb  

if you want just to verify records, python ./read-records.py --type=any.. other then dump  
# training 
python ./train.py  # train using graph  
EAGER=1 python ./train.py # train using eager mode  (using tfrecord)
MODE=valid,test sh ./train/*.sh  just valid and test using eager mode(using tfrecord)  
SHOW=1 sh ./train/*.sh just show model arch in eager mode   

python ./infer.py or INFER=1 sh ./train/*.sh will do infer(without tfrecord) using eager mode    
INFER=1 c0 sh ./train/gru.sh ~/temp/ai2018/sentiment/model/gru/ 
INFER=1 c0 sh ./train/gru.sh ~/temp/ai2018/sentiment/model/gru/ckpt/ckpt-10 
