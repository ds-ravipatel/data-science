# -*- coding: utf-8 -*-
"""
Data set - Download from below location -
https://www.microsoft.com/en-us/download/details.aspx?id=54765&WT.mc_id=rss_alldownloads_devresources
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

DATADIR ="C:/Users/ravip/Documents/Anaconda/Projects/CNNCatVsDog/PetImages"
CATEGORIES = ["Dog","Cat"]

for category in CATEGORIES:   #for each category
    path = os.path.join(DATADIR,category)
    for img in os.listdir(path): #iterate over each image in folder
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) #read image as black and white
        plt.imshow(img_array, cmap = 'gray')
        plt.show()
        break
    break

print(img_array.shape)
'''
So that's a 375 tall, 500 wide, and 3-channel image. 3-channel is because it's RGB (color). 
We definitely don't want the images that big, but also various images are different shapes, 
and this is also a problem.
'''
IMG_SIZE = 100

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))    
plt.imshow(new_array, cmap = 'gray')    
plt.show()


'''Lets Create Training Data'''
'''Take out 15 images from each folder and put under Testing folder'''
training_data = []

def create_training_data():
    for category in CATEGORIES:   #for each category    
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)): #iterate over each image in folder
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) #read image as black and white
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

'''
Next, we want to shuffle the data. Right now our data is just all dogs, then all cats. 
This will usually wind up causing trouble too, as, initially, the classifier will learn 
to just predict dogs always. Then it will shift to oh, just predict all cats! 
Going back and forth like this is no good either.
'''

for sample in training_data[:10]:
    print(sample[1])

import random

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

#below reshaping is required so that it can be passed to model
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
 
#lets save this data -
import pickle

pickle_out = open('X.pickle','wb')
pickle.dump(X, pickle_out)
pickle_out.close()


pickle_out = open('y.pickle','wb')
pickle.dump(y, pickle_out)
pickle_out.close()

import tensorflow as tf

print(tensorflow.__version__)

'''
We will import X and y and train them using CNN
'''
import pickle
import tensorflow as tf

pickle_in = open('X.pickle','rb')
X = pickle.load(pickle_in)

pickle_in = open('y.pickle','rb')
y = pickle.load(pickle_in)

#normalizing X data
X = X/255.0

Xshape1 = X.shape[1:] #we need to pass this as feature shape
Xshape0 = X.shape[0:]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

NAME = "Cats-vs-dogs-64x2-CNN"

tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten()) # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))

model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#checkpoint creation. We will save weights after each epoch
import os
checkpt_path = "Checkpt\cp.ckpt" # this is created in current working dir
checkpt_dir = os.path.dirname(checkpt_path)

import tensorflow as tf

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpt_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3, callbacks=[tensorboard, cp_callback])

#to load weights
model.load_weights(checkpt_path)

model.summary()

#Saving Final Model
model.save('dogVsCatModel.h5')

#loading saved model

from tensorflow.keras import models as keras_models
new_model=keras_models.load_model('dogVsCatModel.h5')

new_model.summary()


# Let's do prediction for few images -
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

DATADIR ="C:/Users/ravip/Documents/Anaconda/Projects/CNNCatVsDog/PetImages/Testing"
CATEGORIES = ["Dog","Cat"]

z = [[1]]

CATEGORIES[np.array(z[0:0]).astype(int)]

#Dog = 0 and Cat = 1

for category in CATEGORIES:   #for each category
    path = os.path.join(DATADIR,category)
    for img in os.listdir(path): #iterate over each image in folder
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) #read image as black and white
        plt.imshow(img_array, cmap = 'gray')
        plt.show()
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        X = np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        #print(X[0:].shape)
        y = np.array(new_model.predict_classes(X)).astype(int)
        print(y)
        #print(y.shape)