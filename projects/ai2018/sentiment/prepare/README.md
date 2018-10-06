first sh ./gen-vocab.sh to get vocab.txt

then using like vocab-mincount.. to get vocab.5k.txt 
then merge with all chars from vocab.txt using vocab-add-char.py to get vocab.5k.chars.txt this is used for word embedding train  

then for canyin and dianping

just use ./gen-mix-seg-canyin.py  ./gen-char-seg-canyin.py ...
not use ./gen-seg-canyin.py for this project for we do not want to consider more words not used here  

do also for dianping and train/valid/test data  

we will use all these data as word embedding pretrain  
