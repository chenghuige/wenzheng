python ./train.py \
    --save_interval_epochs 10 \
    --save_interval_steps 1000  \
    --model_dir /home/gezi/temp/cifar10/model/resnet.2gpu \
    --optimizer momentum \
    --num_gpus=2
