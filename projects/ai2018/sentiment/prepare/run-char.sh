python ./merge-emb.py --emb=./mount/temp/ai2018/sentiment/vectors.char.txt
sh ./gen-valid.sh --seg_method=char
sh ./gen-test.sh --seg_method=char
sh ./gen-train.sh --seg_method=char
