python ./train.py \
    --model_dir=./mount/temp/cifar10/model/resnet \
    --batch_size=256 \
    --save_interval_epochs=5 \
    --metric_eval_interval_steps=0 \
    --valid_interval_epochs=1 \
    --inference_interval_epochs=5 \
    --save_interval_steps 10000
