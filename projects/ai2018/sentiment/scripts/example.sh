#SRC=word.mix.ft c0 sh ./train/v2/torch.mreader.1hop.sh
SRC=word.mix.glove c0 sh ./train/v2/rnet.sh

#SRC=word.tx.ftchar c1 sh ./train/v2/torch.mreader.1hop.wchar.pos.sh  
SRC=word.stanford.glove c1 sh ./train/v2/torch.mreader.1hop.wchar.pos.sh  
#SRC=word.glove c2 sh ./train/v2/torch.mreader.1hop.wchar.pos.sh  
SRC=word.ft c2 sh ./train/v2/torch.mreader.1hop.wchar.pos.sh  
#SRC=word.stanford.tx.ftchar c3 sh ./train/v2/torch.mreader.1hop.wchar.pos.ner.sh
SRC=char.glove c3 sh ./train/v2/torch.mreader.1hop.sh

