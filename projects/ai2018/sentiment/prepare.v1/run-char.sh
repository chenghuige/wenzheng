python ./merge-emb.py 
sh ./gen-valid.sh --seg_method=char
sh ./gen-test.sh --seg_method=char
sh ./gen-train.sh --seg_method=char
