import melt
p = melt.SimplePredictor('./mount/temp/cifar10/model/resnet.momentum.decay/epoch/model.ckpt-30.00-10530', key='pre_logits')
feature = p.inference([melt.read_image('./mount/data/kaggle/cifar-10/test/10.png')])

print(feature)
