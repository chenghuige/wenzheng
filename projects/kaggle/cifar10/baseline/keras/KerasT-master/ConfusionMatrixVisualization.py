import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# Simple example for array bellow

array = [[736,  11,  54,  45,  30,  14,  15 ,  9,  61,  25],
 [ 10, 839,   6 , 38,   3  ,13,   7,   5 , 22,  57],
 [ 47 ,  2 ,566 , 96 ,145 , 65,  51 , 17 ,  7  , 4],
 [ 23  , 6  ,56 ,570  ,97 ,140,  57 , 29 , 12 , 10],
 [ 16  , 2  ,52 , 80 ,700 , 55,  25 , 64 ,  3 ,  3],
 [ 10 ,  1  ,64 ,211 , 59 ,582,  24 , 39 ,  6 ,  4],
 [  4 ,  3  ,42 ,114, 121 , 40, 650 , 13 ,  5 ,  8],
 [ 14 ,  1  ,40 ,57,  69 , 68 , 11 ,723 ,  3  ,14],
 [ 93 , 32  ,26 , 37,  16 , 15 ,  6 ,  2, 752 , 21],
 [ 34 , 83  , 8,  42 , 12  ,21,   6 , 21,  25 ,748]]

df_cm = pd.DataFrame(array, range(10),
                  range(10))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
plt.show()
