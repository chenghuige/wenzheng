#! /bin/bash

mkdir -p ./mount 

#tar czvf afs_mount.tar.gz  /home/HGCP_Program/software-install/afs_mount/

tar xvzf ./hadoop-client.tar.gz
sh ./hadoop-client/hadoop-vfs/bin/mount.sh yq01-build-hdfs.dmop.baidu.com:54310:/user/image-shitu/chenghuige ./mount  > ./mount.out 2>&1 &

ls -l ./mount
sleep 10s
ls -l ./mount

cp ./mount/soft/cuda-9.0.tar.gz . 
tar xvzf cuda-9.0.tar.gz > ./temp.txt 

cp ./mount/soft/py3env.tar.gz .
tar xvzf py3env.tar.gz > ./temp.txt 

cp ./mount/py3env/python.bashrc ./py3env

source ./py3env/python.bashrc

which python 

echo 'pypath'
echo $PYTHONPATH
echo 'libpath'
echo $LD_LIBRARY_PATH

#export LANG=en_US.UTF-8
export PYTHONIOENCODING=utf-8

cd ./test 
python ./test.py
cd ..

tar xvzf wenzheng.tar.gz > ./wenzheng.txt

cd ./wenzheng/projects/ai2018/sentiment
#cd ./wenzheng/projects/ai2018/reader

rm -rf mount 
ln -s ../../../../mount .

export log_dir=`pwd`/../../../log
export TEST_INPUT='1'

#export 'BIG'='1'
#export 'NUM_LAYERS'='3'

export VLOG=1
#export TRAIN_ALL='1'

#export DOUBLE_BATCH='1'

#export BUFFER_SIZE='200000'

export NO_MULTI=1

#supervise=../../../supervise.py
supervise='python ../../../supervise.py'

#which gcc 
#gcc -v 
#strings /lib64/libc.so.6 |grep GLIBC_  

nvidia-smi

export FOLD=0; export SRC=word.ft;sh ./train/v2/torch.mreader.1hop.labelrnn.sh
