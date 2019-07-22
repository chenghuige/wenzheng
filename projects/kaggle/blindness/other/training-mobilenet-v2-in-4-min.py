"""
This kernel takes about 12 min to run:

* Image preprocessing: ~ 8 min
* Model training: ~ 4 min

Since MobileNet v2 has only 2.2M parameters, it trains faster than
ResNet-50 (26M) and ResNext-101 (84M); as a tradeoff, the potential 
accuracy becomes lower. We will finetune the pretrained ImageNet weights.
"""
import os
import json
import math

import cv2
import numpy as np
import pandas as pd
from keras import layers, optimizers
from keras.applications import MobileNetV2
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from sklearn.model_selection import train_test_split 
from tqdm import tqdm


def pad_and_resize(image_path, pad=True, desired_size=224):
    def get_pad_width(im, new_shape, is_rgb=True):
        pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
        t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
        l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
        if is_rgb:
            pad_width = ((t,b), (l,r), (0, 0))
        else:
            pad_width = ((t,b), (l,r))
        return pad_width
        
    img = cv2.imread(image_path)
    
    if pad:
        pad_width = get_pad_width(img, max(img.shape))
        padded = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)
    else:
        padded = img
    
    padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(padded, (desired_size,)*2).astype('uint8')
    
    return resized


def build_model():
    mobilenet = MobileNetV2(
        weights='../input/mobilenet-v2-keras-weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5',
        include_top=False,
        input_shape=(224,224,3)
    )
    model = Sequential()
    model.add(mobilenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='softmax'))
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(lr=0.00002),
        metrics=['accuracy']
    )
    
    return model


# Load Labels
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

# Load Images
train_resized_imgs = []
test_resized_imgs = []

import gezi

t = gezi.Timer('loading train imgs')
for image_id in tqdm(train_df['id_code'], ascii=True):
    img = pad_and_resize(f'../input/aptos2019-blindness-detection/train_images/{image_id}.png')
    train_resized_imgs.append(img)
t.print()

t = gezi.Timer('loading test imgs')
for image_id in tqdm(test_df['id_code'], ascii=True):
    img = pad_and_resize(f'../input/aptos2019-blindness-detection/test_images/{image_id}.png')
    test_resized_imgs.append(img)
t.print()
x_test = np.stack(test_resized_imgs)

print('split data')
# Split Data
x_train, x_val, y_train, y_val = train_test_split(
    np.stack(train_resized_imgs), 
    pd.get_dummies(train_df['diagnosis']).values, 
    test_size=0.1, 
    random_state=2019
)

# Train Model    
model = build_model()
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    verbose=2,
    callbacks=[checkpoint],
    validation_data=(x_val, y_val)
)

# Submission
model.load_weights('model.h5')
y_test = model.predict(x_test)
test_df['diagnosis'] = y_test.argmax(axis=1)
test_df.to_csv('submission.csv',index=False)
