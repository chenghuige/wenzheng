# for single gpu you need to set CUDA... for one specfic gpu otherwise will teat to use multiple gpu(but not horovod)
# run train + eval using horovod (deeep+wide, use index emb, field emb, one mlp  
horovodrun -np 8 sh ./train/test6.sh  # use simple parse can get auc 0.7501
#evaluate only
METRIC=1 sh ./train/test6.sh

#----------some status not important for main purpose
nc python read-test.py --batch_parse=0
nc python read-test.py 
different result 

batch_parse=1 is much faster  

speed much affected by text_dataset.py so if you change some thing there make sure to check speed not down 
