# CIFAR-10 IMAGE CLASSIFICATION WITH KERAS CONVOLUTIONAL NEURAL NETWORK TUTORIAL

## What is Keras?

"Keras is an open source neural network library written in Python and capable of running on top of either [TensorFlow](https://www.tensorflow.org/), [CNTK](https://github.com/Microsoft/CNTK) or [Theano](http://deeplearning.net/software/theano/). 

Use Keras if you need a deep learning libraty that:
* Allows for easy and fast prototyping
* Supports both convolutional networks and recurrent networks, as well as combinations of the two
* Runs seamlessly on CPU and GPU

Keras is compatible with Python 2.7-3.5"[1].

Since Semptember 2016, Keras is the second-fastest growing Deep Learning framework after Google's Tensorflow, and the third largest after Tensorflow and Caffe[2].

## What is Deep Learning?

"Deep Learning is the application to learning tasks of artificial neural networks(ANNs) that contain more than one hidden layer. Deep learning is part of [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning) methods based on learning data representations.
Learning can be [supervised](https://en.wikipedia.org/wiki/Supervised_learning), parially supervised or [unsupervised](https://en.wikipedia.org/wiki/Unsupervised_learning)[3]."

## Project desciption

Simple Youtube presentation what type of visualization is generated:

## What will you learn?




You will learn:

* What is Keras library and how to use it
* What is Deep Learning
* How to use ready datasets
* What is Convolutional Neural Networks(CNN)
* How to build step by step Convolutional Neural Networks(CNN)
* What are differences in model results
* What is supervised and unsupervised learning
* Basics of Machine Learning
* Introduction to Artificial Intelligence(AI)

## Project structure

* 4layerCNN.py - 4-layer Keras model
* 6layerCNN.py - 6-layer Keras model
* README.md - project description step by step

## Convolutional neural network


### 6-layer neural network

#### Network Architecture

```
OPERATION           DATA DIMENSIONS   WEIGHTS(N)   WEIGHTS(%)

               Input   #####      3   32   32
              Conv2D    \|/  -------------------       896     0.0%
                relu   #####     32   32   32
             Dropout    | || -------------------         0     0.0%
                       #####     32   32   32
              Conv2D    \|/  -------------------      9248     0.4%
                relu   #####     32   32   32
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####     32   16   16
              Conv2D    \|/  -------------------     18496     0.8%
                relu   #####     64   16   16
             Dropout    | || -------------------         0     0.0%
                       #####     64   16   16
              Conv2D    \|/  -------------------     36928     1.5%
                relu   #####     64   16   16
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####     64    8    8
              Conv2D    \|/  -------------------     73856     3.1%
                relu   #####    128    8    8
             Dropout    | || -------------------         0     0.0%
                       #####    128    8    8
              Conv2D    \|/  -------------------    147584     6.2%
                relu   #####    128    8    8
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####    128    4    4
             Flatten   ||||| -------------------         0     0.0%
                       #####        2048
             Dropout    | || -------------------         0     0.0%
                       #####        2048
               Dense   XXXXX -------------------   2098176    87.6%
                relu   #####        1024
             Dropout    | || -------------------         0     0.0%
                       #####        1024
               Dense   XXXXX -------------------     10250     0.4%
             softmax   #####          10
```

#### Model
```
model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
    model.add(Dropout(0.2))

    model.add(Conv2D(32,(3,3),padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024,activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))



    sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=False)
```
Train model:
```
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
cnn_n = base_model()
cnn_n.summary()
```
Fit model:
```
cnn = cnn_n.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test,y_test),shuffle=True)
```
#### Results

All results are for 50k iteration, learning rate=0.01. Neural networks have been trained at **16 cores and 16GB RAM** on [plon.io](https://plon.io/)

* epochs = 10 **accuracy=75.61%**

![Keras Training Accuracy vs Validation Accuracy](https://plon.io/files/59651c4b8c5c480001b146f1)
![Keras Training Loss vs Validation Loss](https://plon.io/files/59651c4b8c5c480001b146f3)

**Confusion matrix result:**
```
[[806   9  39  13  28    4   7   9  61  24]
 [ 14 870   4  10   3    4   7   0  28  60]
 [ 69   1 628  64 122   36  44  19  13   4]
 [ 19   5  52 582 109   99  76  29  14  15]
 [ 13   2  44  46 761   27  38  62   6   1]
 [ 15   1  50 189  69  588  31  48   7   2]
 [  8   3  39  53  52   14 814   4  10   3]
 [ 15   3  31  45  63   29   5 795   2  12]
 [ 61  13   8  10  17    1   4   4 875   7]
 [ 23  52  11  10   7    7   5  12  31 842]]
```

**Confusion matrix vizualizing**

![610](https://user-images.githubusercontent.com/11740059/28138320-a88b8e60-6750-11e7-9ec4-ac73ab40ccc7.png)


Time of learning process: **1h 45min**


* epochs = 20 **accuracy=75.31%**


![Keras Training Accuracy vs Validation Accuracy](https://plon.io/files/59654ea78c5c480001b146f9)
![Keras Training Loss vs Validation Loss](https://plon.io/files/59654ea78c5c480001b146fb)

**Confusion matrix result:**
```
[[810   5  30  22  14    2   9  10  60  38]
 [ 13 862   7   8   3    6   4   7  20  70]
 [ 85   2 626  67  84   44  44  27  12   9]
 [ 39   6  47 581  73  137  50  38  17  12]
 [ 22   1  52  87 744   34  22  64   2   2]
 [ 20   3  40 178  44  639  21  48   2   5]
 [ 12   3  42  55  67   16 782  10   7   6]
 [ 15   2  24  38  59   37   3 810   5   7]
 [ 79  14  10  19   6    4   8   5 827  28]
 [ 25  60   8   9   8    5   2  12  21 850]]
```

**Confusion matrix vizualizing**

![620](https://user-images.githubusercontent.com/11740059/28138461-2445946a-6751-11e7-930c-13217c9a5e13.png)


Time of learning process: **3h 40min**

* epochs = 50 **accuracy=69.93%**

![Keras Training Accuracy vs Validation Accuracy](https://plon.io/files/596e67fed81bbc0001085260)
![Keras Training Loss vs Validation Loss](https://plon.io/files/596e67ffd81bbc0001085262)

**Confusion matrix result:**
```
[[760   5  72  32  11   6  12   7  67  28]
 [ 12 862  10  16   3   2  18   4  30  43]
 [ 55   1 712  67  44  35  47  20  11   8]
 [ 37   7 126 554  63  81  69  45  11   7]
 [ 23   2 125  86 622  27  36  69   8   2]
 [ 20   2 121 201  48 488  56  50   7   7]
 [ 16   7 101  65  28  27 734   8  10   4]
 [ 16   4  59  60  57  36   9 749   5   5]
 [107  13  30  32   3  10   8   6 770  21]
 [ 42 100   8  26   8   7   4  21  42 742]]
 ```
 **Confusion matrix vizualizing**
 
![650](https://user-images.githubusercontent.com/11740059/28338609-cf818aa2-6c09-11e7-83c0-89efa60b7c5f.png)


Time of learning process: **8h 10min**


* epochs = 100 **accuracy=68.66%**

![Keras Training Accuracy vs Validation Accuracy](https://plon.io/files/596f6b4bd81bbc0001085268)
![Keras Training Loss vs Validation Loss](https://plon.io/files/596f6b4bd81bbc000108526a)

**Confusion matrix result:**

```
[[736  11  54  45  30  14  15   9  61  25]
 [ 10 839   6  38   3  13   7   5  22  57]
 [ 47   2 566  96 145  65  51  17   7   4]
 [ 23   6  56 570  97 140  57  29  12  10]
 [ 16   2  52  80 700  55  25  64   3   3]
 [ 10   1  64 211  59 582  24  39   6   4]
 [  4   3  42 114 121  40 650  13   5   8]
 [ 14   1  40  57  69  68  11 723   3  14]
 [ 93  32  26  37  16  15   6   2 752  21]
 [ 34  83   8  42  12  21   6  21  25 748]]
```

 **Confusion matrix vizualizing**
 
 ![6100](https://user-images.githubusercontent.com/11740059/28372892-30861cfe-6ca1-11e7-9bbf-83fe0a995c2d.png)


Time of learning process: **17h 10min**

### 4-Layer neural network


#### Network Architecture
```
OPERATION           DATA DIMENSIONS   WEIGHTS(N)   WEIGHTS(%)

               Input   #####      3   32   32
              Conv2D    \|/  -------------------       896     0.1%
                relu   #####     32   32   32
              Conv2D    \|/  -------------------      9248     0.7%
                relu   #####     32   30   30
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####     32   15   15
             Dropout    | || -------------------         0     0.0%
                       #####     32   15   15
              Conv2D    \|/  -------------------     18496     1.5%
                relu   #####     64   15   15
              Conv2D    \|/  -------------------     36928     3.0%
                relu   #####     64   13   13
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####     64    6    6
             Dropout    | || -------------------         0     0.0%
                       #####     64    6    6
             Flatten   ||||| -------------------         0     0.0%
                       #####        2304
               Dense   XXXXX -------------------   1180160    94.3%
                relu   #####         512
             Dropout    | || -------------------         0     0.0%
                       #####         512
               Dense   XXXXX -------------------      5130     0.4%
             softmax   #####          10
```
#### Model

```
model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr = 0.1, decay=1e-6, nesterov=True)
```
Train model:
```
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
cnn_n = base_model()
cnn_n.summary()
```
Fit model:
```
cnn = cnn_n.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test,y_test),shuffle=True)
```

#### Results

All results are for 50k iteration, learning rate=0.1. Neural networks have been trained at **16 cores and 16GB RAM** on [plon.io](https://plon.io/)

* epochs = 10  **accuracy=71.29%**



![Keras Training Accuracy vs Validation Accuracy](https://plon.io/files/5965fdac8c5c48000129f83f)
![Keras Training Loss vs Validation Loss](https://plon.io/files/5965fdac8c5c48000129f841)

**Confusion matrix result:**

```
[[772   5  24  19  17   6  18  10  66  63]
 [ 14 637   1   4   3   7  19   2  38 275]
 [ 81   0 538  50  88  86  93  27  19  18]
 [ 20   1  52 468  60 180 143  33  13  30]
 [ 19   1  51  59 662  33  91  66  16   2]
 [ 12   0  34 135  37 664  53  41  11  13]
 [  7   0  23  29  26  13 885   2  10   5]
 [ 10   0  24  45  48  69  20 756   5  23]
 [ 74   4   3   9   4   6   8   4 854  34]
 [ 18   8   5  13   9   4  10   6  34 893]]
```

**Confusion matrix vizualizing**

![410](https://user-images.githubusercontent.com/11740059/28138095-dd7402c0-674f-11e7-831d-907f59da2723.png)

Time of learning process: **1h 10min**

* epochs = 20 **accuracy=74.57%**



![Keras Training Accuracy vs Validation Accuracy](https://plon.io/files/59639691c0265100013c2c80)
![Keras Training Loss vs Validation Loss](https://plon.io/files/59639691c0265100013c2c82)

**Confusion matrix result:**

```
[[729  11  56  36  52    5  12  11  58  30]
 [  8 883   1  12   5    1  19   0  11  60]
 [ 40   3 545  88 152   54  71  35   7   5]
 [ 10   8  30 583 128  106  75  38  10  12]
 [  7   1  15  37 806    9  43  77   5   0]
 [  6   5  18 214  76  586  32  59   2   2]
 [  3   3  23  58  65    7 825  11   4   1]
 [  5   2  12  56  73   24  11 811   0   6]
 [ 39  30  11  18  18    5  11   5 847  16]
 [ 30  61   3  22   9    2   7   8  16 842]]
```

**Confusion matrix vizualizing**

![420](https://user-images.githubusercontent.com/11740059/28138581-98f7c1c0-6751-11e7-8f87-0d513b29391b.png)

Time of learning process: **2h 15min**

* epochs = 50 **accuracy=75.32%**


![Keras Training Accuracy vs Validation Accuracy](https://plon.io/files/5963e88fc0265100013c2c8c)
![Keras Training Loss vs Validation Loss](https://plon.io/files/5963e890c0265100013c2c8e)




Confusion matrix result:

```
[[727   8  53  25  39    5   4  13  82   44]
 [  7 821   6  10   5    2   4   2  22  121]
 [ 46   0 652  66 107   43  40  30   9    7]
 [ 14   2  61 577 113  125  40  32  13   23]
 [  7   0  46  39 812   15  20  49   6    6]
 [  3   1  47 150  66  649  17  49   4   14]
 [  1   3  49  69  80   27 751   2   9    9]
 [  9   0  22  47  65   38   8 791   3   17]
 [ 32  19   8  21  15    1   6   8 859   31]
 [ 19  30   8  14   5    0   2   9  20 893]]
```

**Confusion matrix vizualizing**

![450](https://user-images.githubusercontent.com/11740059/28140056-0d399676-6757-11e7-976d-29001be11799.png)


Time of learning process: **5h 45min**


* epochs = 100 **accuracy=67.06%**


![Keras Training Accuracy vs Validation Accuracy](https://plon.io/files/5965cefa8c5c480001546c67)
![Keras Training Loss vs Validation Loss](https://plon.io/files/5965cf3e8c5c480001546c69)

Time of learning process: **11h 10min**

Confusion matrix result:

```
[[599   5  74  98  55   14  12   9 117  17]
 [ 16 738  12  65   9   26   7   6  40  81]
 [ 31   0 523 168 136   86  33  14   9   0]
 [ 10   1  31 652  90  175  19  15   5   2]
 [  6   0  34 132 717   55  16  31   9   0]
 [  5   1  17 233  53  661  10  15   4   1]
 [  2   1  39 157 105   48 637   3   7   1]
 [  6   0  14  97 103   96   5 637   5   1]
 [ 41   7  28  84  19   18   6   4 783  10]
 [ 25  28   8  77  29   27   5  19  59 723]]

```


**Confusion matrix vizualizing**

![4100](https://user-images.githubusercontent.com/11740059/28138980-e79bebf2-6752-11e7-9415-e45272435a9c.png)


## Resources

1. [Official Keras Documentation](https://keras.io/)
2. [About Keras on Wikipedia](https://en.wikipedia.org/wiki/Keras)
3. [About Deep Learning on Wikipedia](https://en.wikipedia.org/wiki/Deep_learning)
4. [Tutorial by Dr. Jason Brownlee](http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/)
5. [Tutorial by Parneet Kaur](http://parneetk.github.io/blog/cnn-cifar10/)
6. [Tutorial by Giuseppe Bonaccorso](https://www.bonaccorso.eu/2016/08/06/cifar-10-image-classification-with-keras-convnet/)
7. Open Source on GitHub


## Grab the code or run project in online IDE
* You can [download code from GitHub](https://github.com/simongeek/KerasT)
* You can [run the project in your browser](https://plon.io/explore/keras-cifar-10-classification/m5UMC4nOeCFLM6yXN)
