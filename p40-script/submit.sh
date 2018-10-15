#! /bin/bash

HGCP_CLIENR_BIN=xx

hdfs_dir=/p40/$1
hadoop fs -D hadoop.job.ugi=abc,abc -rmr $hdfs_dir

#rm ./job.env*
#rm ./job.sh.*

${HGCP_CLIENR_BIN}/submit \
    --hdfs hdfs://54310 \
    --hdfs-user abc \
    --hdfs-passwd def \
    --hdfs-path $hdfs_dir \
    --file-dir ./ \
    --job-name $1 \
    --queue-name $2 \
    --num-nodes 1 \
    --num-task-pernode 1 \
    --gpu-pnode $3 \
    --time-limit 0 \
    --job-script ./job.sh 
