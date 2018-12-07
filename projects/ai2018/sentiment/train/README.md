v3 is lstm version of v2 
v4 is same as v2 but decay by f1 

v5 back to using decay by f1 also all add num_finetune_words.. 

v9 refactor move encoder to wenzheng.pyt.BiLMEncoder 
v9 show bert opt better loss for both pyt and tf, and better loss also better f1 for tf, so tf will all using bert opt 

v9 with lm example 

from v11 will use aug mode by default  ?

v12 after contest for ppt re run, but torch elmo wrong input(still improve.. but not as good as using correct one)
v13 fix v12 bug of elmo 

v14 
for jieba best is ./v14/torch.self_attention.unkaug.elmo.finetune_6k.sh  
for sp20w best is ./v14/torch.self_attention.unkaug.elmo.no_finetune_word.sh  
