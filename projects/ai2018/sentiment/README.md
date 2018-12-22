# aichallenger 2018 细粒度用户情感分类第一名解决方案
1. 整体部分参考答辩ppt https://mp.weixin.qq.com/s/W0PhbE8149nD3Venmy33tw
2. 这次比赛客观成绩好的主要原因我觉得是因为我采用了1kw的外部点评数据，这使得我可以采用大词表策略14.4w，19.8w（频次》20）  
   但是开始比赛大部分时间成绩不如do somethig团队，后期取得明显领先优势在于简化版的elmo基于大规模数据的预训练再配合self match attention  
  （这个部分参考rnet是其他选手大部分没有使用的，其实在kaggle toxic我也尝试了但是效果不明显这次比赛它和elmo配合效果非常好）带来单模型效果显著提升。   
  另外我的机器资源相对充分这使得我可以并行跑比如多种分词这种简单但是带来较大模型差异性的模型增强集成效果，这使得我可以使用较少单模型集成取得更好效果。   
3. 其他答辩团队的方案更加新颖细致一些，do something和nevermore团队都采用了attention层独占不共享的策略，这个区别我猜测是我这里前期模型效果不如do something团队的主要原因，待验证。

# 一些注意事项
1. tf 1.10.1  pytorch 1.0  python 3.6  
2. 训练脚本只需要参考 wenzheng/projects/ai2018/sentiment/train/v14 可以参考目录下的readme 我这里提供了一个不利用estimator的bert训练脚本 但是当前bert容易过拟合效果不如rnn  
   单模型bert最优 0.71 rnn 0.72   
3. 需要设置python path export PYTHONPATH=$root/utils:$root 这样可以找到wenzheng/utils/gezi等等lib 
4. 训练的时候 CUDA_VISIBLE_DEVICES=0 或者 CUDA_VISIBLE_DEVICES=0,1(使用两个gpu tower loss) 类似这样 注意pytorch支持多gpu（包括使用buckets length），tensorflow支持多gpu（但是如果采用buckets length，graph模式 不支持多gpu）
   tensorflow理论上支持eager模式 tf的代码我都是基本按照tf.keras接口写的支持eager但是目前tf eager还不完善。  
5. 当前最好的单模型torch版本（torch比tensorflow占用显存更大）如果使用1080ti 11g显存 需要2个gpu跑，p40可以单卡跑。 当然你也可以调小batch size 或者调整 buckets以及batch sizes  

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
