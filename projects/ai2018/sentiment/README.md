# ai challenger 2018 细粒度用户评论情感分类第一名（后厂村静静团队）解决方案
1. 整体部分参考答辩ppt https://mp.weixin.qq.com/s/W0PhbE8149nD3Venmy33tw  
2. 这次比赛客观成绩好的主要原因我觉得是因为我从比赛一开始就采用了1kw的外部点评数据，这使得我可以采用大词表策略14.4w，19.8w（频次》20） 
   但是比赛大部分时间成绩不如do somethig团队，后期取得明显领先优势应该主要是简化版的elmo基于大规模数据的预训练再配合self match attention（self match attention参考rnet是其他选手大部分没有使用的，其实在kaggle toxic我也尝试了但是效果不明显这次比赛它和elmo配合效果非常好）带来单模型效果显著提升。 
  fast elmo部分可以参考 torch-lm-train.py tf版本对应 lm-train.py 但是基本相同的实现和完全相同的训练数据 tf版本可能训练过程中经常出现NAN，待修复 pytorch版本训练稳定。注意目前的设计只能单层elmo如果多层需要再修改代码因为目前是stack rnn第二层就等于loss作弊了(不过单层elmo预训练在这个数据效果足够好)  
  我的机器资源相对充分这使得我可以并行跑比如多种分词这种简单但是带来较大模型差异性的模型增强集成效果，这样我可以使用较少单模型集成取得更好效果。   
3. 其他答辩团队的方案更加新颖细致一些，受益匪浅。do something和nevermore团队都采用了attention pooling层独占不共享的策略，而我采用的pooling层完全共享(topk concat attention)  
   pooling层的独占策略使得模型更加具有可解释性。 另外do somthing团队使用了sru也值得尝试。  
   new start团队采用了seq2seq的模型 并且使用了平衡准召的loss nevermore团队对稀有类别过采样处理 do somthing团队做了标注纠正 并且引入了基于词汇和类别共现的特征表示。  
4. 关于pooling层独占的实验效果  
   赛后在当前最佳单模型基础上实验了pooling层独占效果,f1直接指标提高比较明显，有一个exclusive_fc取得了最优loss，考虑模型随机性提升不是特别明显，当然更多调参可能还能提升一下效果，更主要的好处还是带来更好的模型可解释性。    
   sh ./train/v14/torch.self_attention.unkaug.elmo.finetune_6k.exclusive_pooling.exclusive_fc.sh              adjusted_f1 0.7193  f1 0.7144 loss 0.3215  
   sh ./train/v14/torch.self_attention.unkaug.elmo.finetune_6k.exclusive_pooling.exclusive_fc.onlyatt.sh      adjusted_f1 0.7207  f1 0.7145 loss 0.3236  

# 一些注意事项
1. python 3.6 tf 1.10.1  pytorch 1.0  即使只使用基于pytorch的模型也依赖tensorflow因为采用了tf dataset读取tfrecord(对应pytorch的时候使用的是tf的eager模式读取数据)  
2. 训练脚本只需要参考 wenzheng/projects/ai2018/sentiment/train/v14 可以参考train目录下的readme 我这里提供了一个不利用estimator的bert训练脚本 但是当前bert容易过拟合效果不如rnn  
   单模型bert最优 0.71(train/v14/tf.char.transformer.nbert.finetune.4gpu.3epoch.sh)   
   rnn 0.72(train/v14/torch.self_attention.unkaug.elmo.finetune_6k.sh) 比赛中的最佳单模型 建议只参考基于pytorch rnn的这个最佳单模型,  
   或者赛后实验的pooling层独占(train/v14/torch.self_attention.unkaug.elmo.finetune_6k.exclusive_pooling.exclusive_fc.onlyatt.sh)  
3. 需要设置python path export PYTHONPATH=$root/utils:$root 这样可以找到wenzheng/utils/gezi等等lib 
4. 训练的时候 CUDA_VISIBLE_DEVICES=0 sh ./train/v14/a.sh 或者 CUDA_VISIBLE_DEVICES=0,1 sh ./train/v14/a.sh(使用两个gpu tower loss) 类似这样 注意pytorch支持多gpu（包括使用buckets length），tensorflow支持多gpu（但是如果采用buckets length，graph模式 不支持多gpu）  
   tensorflow理论上支持eager模式 tf的代码我都是基本按照tf.keras接口写的支持eager但是目前tf eager还不完善。  
5. 当前最好的单模型torch版本（torch比tensorflow占用显存更大）如果使用1080ti 11g显存 需要2个gpu跑，p40可以单卡跑。 当然你也可以调小batch size 或者调整 buckets以及batch sizes   
6. 模型集成只需要参考 ensemble/ensemble-cv.py 我这里利用了valid数据做交叉验证，没有使用全部数据多fold的方式（那样模型训练代价比较高）   

# 生成好的tfrecords
为了方便训练这里提供了预先处理好的tfrecords可以在百度云盘下载  
链接: https://pan.baidu.com/s/1Ehixs1gbS4nv4a1ivI6TwA 提取码: rzd1   
包括最佳单模型对应tfrecords word.jieba.ft 对应elmo模型 model/lm/.../latest.pyt  

# generate tfrecord  
go to prepare, sh run.sh gen valid/test/train   
# dump infos.pkl  
python ./read-records.py --type dump   

or just sh prepare.sh to do above, be sure you use correct tfrecord folder (using ln -s)  and vectors.txt if use word pretrain emb  

if you want just to verify records, python ./read-records.py --type=any.. other then dump  
# training 
python ./train.py  # train using graph  
python ./torch-train.py # train using eager tf + pytorch  
EAGER=1 python ./train.py # train using eager mode  (using tfrecord)
MODE=valid,test sh ./train/*.sh  just valid and test using eager mode(using tfrecord)  
SHOW=1 sh ./train/*.sh just show model arch in eager mode   

python ./infer.py or INFER=1 sh ./train/*.sh will do infer(without tfrecord) using eager mode    
INFER=1 c0 sh ./train/gru.sh ~/temp/ai2018/sentiment/model/gru/ 
INFER=1 c0 sh ./train/gru.sh ~/temp/ai2018/sentiment/model/gru/ckpt/ckpt-10 
