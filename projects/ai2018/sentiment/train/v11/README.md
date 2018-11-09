-----tf 
rnet.nolatt 
rnet 

lm
ft 
glove 

jieba 
sp10w 
sp1w
mix 
char 

hidden 400
padding 
concat layers 
reccrent dropout 
gate attention 

transformer (bert)
transformer (bert) + gru

-----torch   
mreader.nolatt 
mreader

bertopt + unkaug 
adamax + loss decay 

hidden 768
no padding 
not concat layers
simple dropout 
sfu attention 
