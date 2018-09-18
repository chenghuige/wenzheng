# generate tfrecord  
go to prepare, sh run.sh gen valid/test/train   
# dump infos.pkl  
python ./read-records.py --type dump   

or just sh prepare.sh to do above, be sure you use correct tfrecord folder (using ln -s)  and vectors.txt if use word pretrain emb  

if you want just to verify records, python ./read-records.py --type=any.. other then dump  
# training 
python ./train.py  # train using graph  
EAGER=1 python ./train.py # train using eager mode   
MODE=valid,test sh ./train/*.sh  just valid and test using eager mode  
SHOWMODEL=1 sh ./train/*.sh just show model arch in eager mode  
