py run.py qiano  

v3 use lstm cell
v4 same as v3 but decay by adjusted_f1 not f1  

v4 also for weights decay use decay factor 0.9   

v5 same as v4 but decay by loss , decay factor 0.5 for weights decay

v6 same as v5 but decay by auc  


v7 word + char gru decay start 3
v8 decay start 2 
v9 decay start 1
v10 decay start 2, lstm 
v11 decay start 1, lstm 
v12 same as v11 but sfu combine word and char
