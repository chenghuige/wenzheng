# infer for tf models just use graph mode for safe
MODE=test INFER=1 sh ./infer/v11/tf.word.rnetv2.lm.sh /home/gezi/temp/ai2018/sentiment/model/v11/0/word.jieba.ft/tf.word.rnetv2.gru.lm/epoch/model.ckpt-5.00-16410 
# infer for pytorch models  
MODE=test INFER=1 sh ./train/v10/torch.word.mreader.nopad.lm.unkaug.sh /home/gezi/temp/ai2018/sentiment/model/v10/0/word.jieba.ft/torch.word.mreader.nopad.lm.gru.unkaug/ckpt/ckpt-6 
