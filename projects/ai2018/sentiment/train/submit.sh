#! /bin/bash

#HGCP_CLIENR_BIN=/home/users/chenghuige/.hgcp/software-install/HGCP_client/bin
HGCP_CLIENR_BIN=~/.hgcp/software-install/HGCP_client/bin
#HGCP_CLIENR_BIN=/home/chenghuige/.hgcp/software-install/HGCP_client/bin

hdfs_dir=/user/image-shitu/chenghuige/p40/$1
hadoop fs -D hadoop.job.ugi=image-shitu,image-shitu_build -rmr $hdfs_dir

#rm ./job.env*
#rm ./job.sh.*

#--queue-name yq01-p40-$2 \
# --time-limit 86400 \
${HGCP_CLIENR_BIN}/submit \
    --hdfs hdfs://yq01-build-hdfs.dmop.baidu.com:54310 \
    --hdfs-user image-shitu \
    --hdfs-passwd image-shitu_build \
    --hdfs-path $hdfs_dir \
    --file-dir ./ \
    --job-name $1 \
    --queue-name $2 \
    --num-nodes 1 \
    --num-task-pernode 1 \
    --gpu-pnode $3 \
    --time-limit 0 \
    --job-script ./job.sh 
